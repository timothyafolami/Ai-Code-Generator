import boto3
import os
import json
from typing import Optional, List, Dict, Any
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import asyncio

load_dotenv()

class ChatBedrock:
    """
    A class to handle AWS Bedrock conversations in a similar way to ChatAnthropic.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        max_tokens: int = 20000,
        temperature: float = 0.1,
    ):
        """
        Initialize the ChatBedrock client.

        Args:
            model_id: Bedrock model ID or inference profile ID
            region_name: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
        """
        self.model_id = model_id or os.getenv("BEDROCK_INFERENCE_PROFILE_ID", "anthropic.claude-3-sonnet-20240229-v1:0") # Default to Sonnet if not set
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Basic config validation
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials required in environment variables")

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

    def invoke(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Invoke the Bedrock model with a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for inference config

        Returns:
            Response text from the model
        """
        try:
            inference_config = {
                "maxTokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            formatted_messages: List[Dict[str, Any]] = []
            system_blocks: List[Dict[str, Any]] = []

            for msg in messages:
                # LangChain-style messages with .type/.content
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role = "user" if getattr(msg, "type", "").lower() in ("human", "user") else "assistant"
                    formatted_messages.append({"role": role, "content": [{"text": str(getattr(msg, "content", ""))}]})
                    continue

                # Dict-style {role, content}
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = str(msg.get("role", "user")).lower()
                    content = msg.get("content")
                    # Normalize content to Bedrock blocks
                    if isinstance(content, list):
                        blocks = content
                    else:
                        blocks = [{"text": str(content)}]

                    if role == "system":
                        system_blocks.extend(blocks)
                    elif role in ("user", "assistant"):
                        formatted_messages.append({"role": role, "content": blocks})
                    else:
                        # Fallback: treat unknown roles as user
                        formatted_messages.append({"role": "user", "content": blocks})
                    continue

                raise ValueError(f"Unsupported message format in invoke: {msg}")

            req = {
                "modelId": self.model_id,
                "messages": formatted_messages,
                "inferenceConfig": inference_config,
            }
            if system_blocks:
                req["system"] = system_blocks

            response = self.client.converse(**req)

            return response["output"]["message"]["content"][0]["text"]
        
        except ClientError as e:
            raise Exception(f"Bedrock API error: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected response format: {e}")
        except Exception as e:
            raise Exception(f"Error invoking Bedrock model: {e}")

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple chat interface similar to other chat implementations.
        This method will continue to work well with string inputs.

        Args:
            message: User message
            system_prompt: Optional system prompt for context

        Returns:
            Response text from the model
        """
        try:
            messages = []

            if system_prompt:
                messages.append(
                    {"role": "user", "content": [{"text": f"{system_prompt}\n\n{message}"}]}
                )
            else:
                messages.append({"role": "user", "content": [{"text": message}]})

            return self.invoke(messages)
        
        except Exception as e:
            raise Exception(f"Error in chat method: {e}")


class BedrockLLM(Runnable):
    """
    LangChain-compatible wrapper for ChatBedrock that works in both standalone and chain modes.
    """

    def __init__(self, parser=None, **kwargs):
        super().__init__()
        self.chat_bedrock = ChatBedrock(**kwargs)
        self.parser = parser or StrOutputParser()
    
    def invoke(self, input_data, config=None) -> str:
        """
        Invoke method that always returns raw strings (for LangChain compatibility).

        Args:
            input_data: String, StringPromptValue, or dict with template/variables, or list of messages
            config: LangChain config (ignored)

        Returns:
            Raw string response (let chains handle parsing)
        """
        try:
            if isinstance(input_data, str):
                response = self.chat_bedrock.chat(input_data)
            elif hasattr(input_data, "to_string"):  # LangChain StringPromptValue
                prompt_text = input_data.to_string()
                response = self.chat_bedrock.chat(prompt_text)
            elif isinstance(input_data, dict) and "template" in input_data:  # Our custom dict format
                template = input_data.get("template", "")
                variables = input_data.get("variables", {})
                formatted_prompt = template.format(**variables)
                response = self.chat_bedrock.chat(formatted_prompt)
            elif isinstance(input_data, list):  # Expected for LangChain HumanMessage, AIMessage etc.
                response = self.chat_bedrock.invoke(input_data)
            else:
                raise ValueError(f"Unsupported input format: {type(input_data)}")

            # Parse through the configured parser (JSON or string)
            return self.parser.parse(response)
        
        except Exception as e:
            raise Exception(f"Error in BedrockLLM invoke: {e}")

    async def ainvoke(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Asynchronous invoke method for LangChain compatibility.
        Delegates to the synchronous invoke and wraps in asyncio.
        """
        try:
            return await asyncio.to_thread(self.invoke, input_data, config)
        except Exception as e:
            raise Exception(f"Error in async invoke: {e}")

    def invoke_with_parser(self, input_data) -> Any:
        """
        Invoke with parsing for standalone use.
        """
        try:
            response = self.invoke(input_data)
            return self.parser.parse(response)
        except Exception as e:
            raise Exception(f"Error in invoke_with_parser: {e}")


class LLMController:
    # Removed openai_api_key and claude_api_key from init, as only Bedrock is used.
    def __init__(self):
        pass

    def aws_bedrock(self) -> BedrockLLM:
        """Call the AWS Bedrock model with the given query."""
        # Temperature is now passed directly to BedrockLLM which passes to ChatBedrock
        return BedrockLLM(parser=StrOutputParser(), temperature=0.0)

    def aws_bedrock_json(self) -> BedrockLLM:
        """Call the AWS Bedrock model with the given query and return the response in JSON format."""
        # Temperature is now passed directly to BedrockLLM which passes to ChatBedrock
        return BedrockLLM(parser=JsonOutputParser(), temperature=0.0)


def get_llm(model_family: str = "aws_bedrock") -> BedrockLLM: # Changed default to "aws_bedrock"
    """Get the appropriate LLM (now only Bedrock)."""
    try:
        controller = LLMController() # No longer needs API keys here
        return controller.aws_bedrock()
    except Exception as e:
        raise Exception(f"Error initializing LLM: {e}")


def get_llm_json(model_family: str = "aws_bedrock") -> BedrockLLM: # Changed default to "aws_bedrock"
    """Get the appropriate JSON LLM (now only Bedrock)."""
    try:
        controller = LLMController() 
        return controller.aws_bedrock_json()
    except Exception as e:
        raise Exception(f"Error initializing JSON LLM: {e}")

# RE-ADDED: Global LLM and LLM_JSON instances, initialized once here
claude_llm_json = get_llm_json()
claude_llm = get_llm()
