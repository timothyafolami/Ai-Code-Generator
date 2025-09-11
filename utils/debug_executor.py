# =============================================================================
# DEBUG CODE EXECUTOR - Find and Fix Issues
# =============================================================================

import sys
import time
import traceback
from io import StringIO
import gc

class DebugDataAnalysisExecutor:
    """Simplified version for debugging"""
    
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 
            'scipy', 'datetime', 'time', 'json', 're', 'math', 
            'statistics', 'typing', 'warnings'
        }
        self.data_cache = {}
    
    def execute_with_context(self, code, preserve_context=True):
        """Execute code with detailed debugging"""
        
        print(f"🔍 DEBUG: Executing code:")
        print("=" * 40)
        print(code)
        print("=" * 40)
        
        # Setup execution environment
        execution_globals = self._create_safe_globals()
        execution_locals = self.data_cache.copy() if preserve_context else {}
        
        print(f"🔍 DEBUG: Available variables before execution: {list(execution_locals.keys())}")
        
        # Setup output capture
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'locals': {}
        }
        
        start_time = time.time()
        
        try:
            # Redirect output
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            
            print(f"🔍 DEBUG: About to execute code...")
            
            # Execute the code
            exec(code, execution_globals, execution_locals)
            
            result['success'] = True
            result['locals'] = self._filter_locals(execution_locals)
            result['execution_time'] = time.time() - start_time
            
            print(f"🔍 DEBUG: Code executed successfully!")
            
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {str(e)}'
            result['traceback'] = traceback.format_exc()
            result['execution_time'] = time.time() - start_time
            
            print(f"🔍 DEBUG: Code execution failed!")
            print(f"🔍 DEBUG: Error: {result['error']}")
            print(f"🔍 DEBUG: Full traceback:")
            print(result['traceback'])
            
        finally:
            # Restore output
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
            # Capture outputs
            result['output'] = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            if stderr_output and not result['error']:
                result['error'] = stderr_output
        
        # Update cache if successful
        if result['success'] and preserve_context:
            self.data_cache.update(result['locals'])
            print(f"🔍 DEBUG: Updated cache with variables: {list(result['locals'].keys())}")
        
        print(f"🔍 DEBUG: Final result: success={result['success']}")
        print(f"🔍 DEBUG: Variables in cache: {list(self.data_cache.keys())}")
        
        return result
    
    def _create_safe_globals(self):
        """Create safe globals with debugging"""
        
        print("🔍 DEBUG: Creating safe globals...")
        
        # Use full builtins but filter dangerous ones
        import builtins
        safe_builtins = {}
        
        dangerous_functions = ['exec', 'eval', 'compile', 'open']
        
        for name in dir(builtins):
            if not name.startswith('_') and name not in dangerous_functions:
                safe_builtins[name] = getattr(builtins, name)
        
        print(f"🔍 DEBUG: Safe builtins count: {len(safe_builtins)}")
        
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split('.')[0]
            print(f"🔍 DEBUG: Attempting to import '{name}' (base: '{base_module}')")
            
            if base_module not in self.allowed_modules:
                error_msg = f"Import of '{name}' is not allowed. Allowed modules: {self.allowed_modules}"
                print(f"🔍 DEBUG: Import blocked: {error_msg}")
                raise ImportError(error_msg)
            
            try:
                imported = __import__(name, globals, locals, fromlist, level)
                print(f"🔍 DEBUG: Successfully imported '{name}'")
                return imported
            except Exception as e:
                print(f"🔍 DEBUG: Import failed for '{name}': {e}")
                raise
        
        globals_dict = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
        }
        
        print(f"🔍 DEBUG: Created globals with keys: {list(globals_dict.keys())}")
        
        return globals_dict
    
    def _filter_locals(self, locals_dict):
        """Filter locals with debugging"""
        
        print(f"🔍 DEBUG: Filtering locals from: {list(locals_dict.keys())}")
        
        filtered = {}
        for key, value in locals_dict.items():
            if key.startswith('_'):
                print(f"🔍 DEBUG: Skipping private variable: {key}")
                continue
            
            import types
            if isinstance(value, types.ModuleType):
                print(f"🔍 DEBUG: Skipping module: {key}")
                continue
            
            if isinstance(value, types.FunctionType):
                print(f"🔍 DEBUG: Skipping function: {key}")
                continue
            
            try:
                str(value)  # Test if serializable
                filtered[key] = value
                print(f"🔍 DEBUG: Kept variable: {key} = {type(value)}")
            except Exception as e:
                filtered[key] = f"<{type(value).__name__} object>"
                print(f"🔍 DEBUG: Variable {key} not serializable: {e}")
        
        print(f"🔍 DEBUG: Filtered locals: {list(filtered.keys())}")
        return filtered
    
    def list_variables(self):
        return list(self.data_cache.keys())
    
    def get_variable(self, var_name):
        return self.data_cache.get(var_name)
    
    def clear_context(self):
        self.data_cache.clear()

# =============================================================================
# SIMPLE TEST
# =============================================================================

def test_debug_executor():
    """Test with detailed debugging"""
    
    print("🚀 STARTING DEBUG TEST")
    print("=" * 50)
    
    executor = DebugDataAnalysisExecutor()
    
    # Test 1: Simple test first
    print("\n📝 TEST 1: Simple Python code")
    simple_code = """
x = 42
y = "hello"
print(f"x = {x}, y = {y}")
result = x * 2
print(f"result = {result}")
"""
    
    result1 = executor.execute_with_context(simple_code)
    print(f"✅ Result 1: success={result1['success']}")
    if result1['success']:
        print(f"Output: {result1['output']}")
    else:
        print(f"Error: {result1['error']}")
    
    # Test 2: Import numpy
    print("\n📝 TEST 2: Import numpy")
    numpy_code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
"""
    
    result2 = executor.execute_with_context(numpy_code)
    print(f"✅ Result 2: success={result2['success']}")
    if result2['success']:
        print(f"Output: {result2['output']}")
    else:
        print(f"Error: {result2['error']}")
    
    # Test 3: Import pandas
    print("\n📝 TEST 3: Import pandas")
    pandas_code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print("DataFrame created:")
print(df)
print(f"Shape: {df.shape}")
"""
    
    result3 = executor.execute_with_context(pandas_code)
    print(f"✅ Result 3: success={result3['success']}")
    if result3['success']:
        print(f"Output: {result3['output']}")
    else:
        print(f"Error: {result3['error']}")
    
    # Test 4: Use previous context
    print("\n📝 TEST 4: Use previous context")
    context_code = """
# Use variables from previous execution
if 'df' in locals() or 'df' in globals():
    print("DataFrame found!")
    print(f"Sum of all values: {df.sum().sum()}")
else:
    print("DataFrame not found in context")
    print("Available variables:", [k for k in locals().keys() if not k.startswith('_')])
"""
    
    result4 = executor.execute_with_context(context_code)
    print(f"✅ Result 4: success={result4['success']}")
    if result4['success']:
        print(f"Output: {result4['output']}")
    else:
        print(f"Error: {result4['error']}")
    
    # Show final state
    print(f"\n🏁 FINAL STATE:")
    print(f"Variables in cache: {executor.list_variables()}")

if __name__ == "__main__":
    test_debug_executor()