# backend/utils/code_validator.py
import re
import ast
from typing import Tuple


class CodeValidator:
    """
    Validates generated Python code for safety and correctness before execution.
    """

    # Dangerous patterns that should not appear in generated code
    FORBIDDEN_PATTERNS = [
        r'\bos\s*\.',          # os module calls
        r'\bsys\s*\.',         # sys module calls
        r'\b__import__\s*\(',  # dynamic imports
        r'\beval\s*\(',        # eval
        r'\bexec\s*\(',        # exec
        r'\bopen\s*\(',        # file open (except FILE_PATH reading)
        r'\bsubprocess\s*\.',  # subprocess
        r'\bsocket\s*\.',      # networking
        r'\brequests\s*\.',    # HTTP requests
        r'\bimport\s+(?!pandas|numpy|matplotlib)', # imports other than allowed
        r'\bfrom\s+(?!pandas|numpy|matplotlib)', # from imports other than allowed
    ]

    # Required patterns for valid pandas code
    REQUIRED_PATTERNS = [
        r'RESULT\s*=',  # Must assign to RESULT
    ]

    @staticmethod
    def validate(code: str) -> Tuple[bool, str]:
        """
        Validate generated code for syntax, safety, and correctness.
        
        Returns:
            (is_valid, error_message_or_empty_string)
        """
        if not code or not isinstance(code, str):
            return False, "Code is empty or not a string"

        code_stripped = code.strip()
        if not code_stripped:
            return False, "Code contains only whitespace"

        # Check code length (prevent huge generations)
        if len(code) > 15000:  # Increased from 10000 to allow more complex visualizations
            return False, "Code too long (max 15000 characters)"

        # Check for forbidden patterns (security check)
        for pattern in CodeValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"

        # Check for required patterns - must have RESULT assignment
        if not re.search(r'RESULT\s*=', code):
            return False, "Code must assign to RESULT variable (e.g., RESULT = plt or RESULT = df)"

        # Try to parse as valid Python
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            error_msg = str(e)
            if 'unexpected EOF' in error_msg or 'invalid syntax' in error_msg:
                return False, f"Syntax error: {error_msg}. Use semicolons to separate statements on ONE line."
            return False, f"Syntax error: {error_msg}"

        # Check for infinite loops
        if CodeValidator._has_infinite_loop(code):
            return False, "Code may contain infinite loops"

        return True, ""

    @staticmethod
    def _has_infinite_loop(code: str) -> bool:
        """
        Simple heuristic to detect potential infinite loops.
        Looks for while True with no break or return.
        """
        # Look for while True
        if not re.search(r'\bwhile\s+True\s*:', code):
            return False

        # Check if there's a break or return after it
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    # Check if the test is a constant True
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        # Check if body has break or return
                        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                        if not has_break and not has_return:
                            return True
        except Exception:
            pass

        return False

    @staticmethod
    def sanitize_code_for_logging(code: str, max_length: int = 500) -> str:
        """
        Create a safe version of code for logging (truncated).
        """
        if len(code) > max_length:
            return code[:max_length] + "...[truncated]"
        return code
