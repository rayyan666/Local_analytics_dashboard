"""
Error Suggestion Engine - Provides intelligent error messages with helpful hints
"""
import re
from typing import Dict, Tuple

class ErrorSuggester:
    """
    Analyzes error messages and provides helpful suggestions for common data analysis issues.
    """
    
    # Pattern matching for common errors with helpful suggestions
    ERROR_PATTERNS = {
        r"KeyError.*?(['\"])([^'\"]+)": {
            "hint": "Column '{column}' does not exist in the dataset",
            "suggestion": "Try asking 'What columns do I have?' to see available columns"
        },
        r"NameError.*?name '([^']+)' is not defined": {
            "hint": "Variable '{variable}' is not available",
            "suggestion": "Make sure to define all variables before using them"
        },
        r"AttributeError.*?has no attribute '([^']+)'": {
            "hint": "Object doesn't have attribute '{attribute}'",
            "suggestion": "Check the correct method/property name for this operation"
        },
        r"ValueError.*?could not convert string to float": {
            "hint": "A column contains non-numeric values where numbers are expected",
            "suggestion": "Try using .dropna() or convert the column to numeric with pd.to_numeric()"
        },
        r"TypeError.*?unsupported operand type": {
            "hint": "Operation attempted on incompatible data types",
            "suggestion": "Check your data types - convert strings to numbers if needed with astype()"
        },
        r"timeout|Killed|memory": {
            "hint": "Query took too long or used too much memory",
            "suggestion": "Try limiting data: df.head(100) or use sampling: df.sample(n=1000)"
        },
        r"FileNotFoundError|No such file": {
            "hint": "File path is incorrect or file doesn't exist",
            "suggestion": "Verify the file path is correct and the file has been uploaded"
        },
        r"IndexError|index out of range": {
            "hint": "Trying to access an element that doesn't exist",
            "suggestion": "Check if your data has enough rows/columns before accessing by index"
        },
        r"ZeroDivisionError": {
            "hint": "Division by zero detected",
            "suggestion": "Add a check to avoid dividing by zero: df = df[df.column != 0]"
        },
        r"AssertionError": {
            "hint": "An assertion in the code failed",
            "suggestion": "Check if your data meets expected conditions"
        },
    }
    
    EXECUTION_ERROR_PATTERNS = {
        r"Empty DataFrame": {
            "hint": "Your filter resulted in no rows",
            "suggestion": "Check your filter conditions - they may be too restrictive"
        },
        r"No column named": {
            "hint": "Column doesn't exist after normalization",
            "suggestion": "Column names are normalized to lowercase. Check available columns."
        },
        r"cannot access property|cannot read|undefined": {
            "hint": "Trying to access a property that doesn't exist",
            "suggestion": "Verify the property name and object structure"
        },
    }
    
    @staticmethod
    def suggest(error_message: str) -> Dict[str, str]:
        """
        Analyzes an error message and returns helpful suggestions.
        
        Args:
            error_message: The error message to analyze
            
        Returns:
            Dictionary with 'message', 'hint', 'suggestion', and 'code_fix' keys
        """
        error_lower = error_message.lower()
        
        # Try to match against known patterns
        for pattern, suggestion_template in ErrorSuggester.ERROR_PATTERNS.items():
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                hint = suggestion_template["hint"]
                suggestion = suggestion_template["suggestion"]
                
                # Extract captured groups for dynamic hints
                if match.groups():
                    try:
                        # For KeyError pattern: extract column name
                        if "column" in hint:
                            column = match.group(2) if len(match.groups()) >= 2 else match.group(1)
                            hint = hint.format(column=column)
                        # For NameError/AttributeError patterns
                        elif "variable" in hint:
                            hint = hint.format(variable=match.group(1))
                        elif "attribute" in hint:
                            hint = hint.format(attribute=match.group(1))
                    except (IndexError, KeyError):
                        pass
                
                return {
                    "message": error_message.split('\n')[0],  # First line only
                    "hint": hint,
                    "suggestion": suggestion,
                    "code_fix": ErrorSuggester._suggest_code_fix(error_message, pattern)
                }
        
        # Try execution-specific patterns
        for pattern, suggestion_template in ErrorSuggester.EXECUTION_ERROR_PATTERNS.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                return {
                    "message": error_message.split('\n')[0],
                    "hint": suggestion_template["hint"],
                    "suggestion": suggestion_template["suggestion"],
                    "code_fix": ""
                }
        
        # Default response for unknown errors
        return {
            "message": error_message.split('\n')[0],
            "hint": "An unexpected error occurred",
            "suggestion": "Try a simpler query or check your data format",
            "code_fix": ""
        }
    
    @staticmethod
    def _suggest_code_fix(error_message: str, pattern: str) -> str:
        """
        Suggests a code fix based on the error type.
        
        Args:
            error_message: The error message
            pattern: The regex pattern that matched
            
        Returns:
            A suggested code snippet to fix the issue
        """
        if "KeyError" in pattern:
            return "# Check available columns:\ndf.columns.tolist()\n# Then use the correct column name"
        elif "NameError" in pattern:
            return "# Make sure to define the variable first\nvariable = value"
        elif "ValueError" in pattern and "float" in error_message:
            return "# Convert to numeric, handling non-numeric values:\ndf['column'] = pd.to_numeric(df['column'], errors='coerce')"
        elif "timeout" in error_message.lower():
            return "# Limit the dataset:\ndf = df.head(100)\n# or sample:\ndf = df.sample(n=1000)"
        elif "ZeroDivisionError" in pattern:
            return "# Avoid division by zero:\nresult = df[df.denominator != 0].apply(lambda x: x['numerator'] / x['denominator'])"
        
        return ""
    
    @staticmethod
    def format_helpful_response(error_msg: str, context: str = "") -> str:
        """
        Formats a helpful error response for the user.
        
        Args:
            error_msg: The error message
            context: Additional context (like column names available)
            
        Returns:
            A formatted, user-friendly error message
        """
        suggestion = ErrorSuggester.suggest(error_msg)
        
        formatted = f"❌ **{suggestion['hint']}**\n\n"
        formatted += f"📝 {suggestion['suggestion']}"
        
        if suggestion['code_fix']:
            formatted += f"\n\n💡 **Example fix:**\n```python\n{suggestion['code_fix']}\n```"
        
        if context:
            formatted += f"\n\n📊 {context}"
        
        return formatted
