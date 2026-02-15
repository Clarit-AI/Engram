"""
Tool call parser for extracting tool calls from model outputs.

Supports multiple formats:
- OpenAI function calling format
- XML-based tool calls
- JSON tool calls
- Custom formats
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

# Attempt to import defusedxml for safe XML parsing
try:
    import defusedxml.ElementTree as DefusedET
    DEFUSEDXML_AVAILABLE = True
except ImportError:
    DEFUSEDXML_AVAILABLE = False
    # Note: Will warn when XML parsing is attempted

logger = logging.getLogger(__name__)


class ToolCallFormat(Enum):
    """Supported tool call formats."""

    OPENAI_FUNCTION = "openai_function"  # OpenAI function calling
    XML = "xml"  # <tool_call><name>...</name><parameters>...</parameters></tool_call>
    JSON = "json"  # {"tool": "...", "parameters": {...}}
    MARKDOWN_JSON = "markdown_json"  # ```json\n{...}\n```
    AUTO = "auto"  # Auto-detect format


@dataclass
class ParsedToolCall:
    """
    Parsed tool call.

    Attributes:
        name: Tool name
        parameters: Tool parameters
        raw_text: Original text that was parsed
        format: Format used for parsing
        confidence: Confidence score (0.0-1.0)
    """

    name: str
    parameters: Dict[str, Any]
    raw_text: str
    format: ToolCallFormat
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "raw_text": self.raw_text,
            "format": self.format.value,
            "confidence": self.confidence,
        }


class ToolCallParser:
    """
    Parser for extracting tool calls from model outputs.

    This parser supports multiple tool call formats and can auto-detect
    the format used in the model output.

    **Supported Formats:**

    1. OpenAI Function Calling:
       {
           "name": "tool_name",
           "arguments": "{\"param1\": \"value1\"}"
       }

    2. XML:
       <tool_call>
           <name>tool_name</name>
           <parameters>
               <param1>value1</param1>
           </parameters>
       </tool_call>

    3. JSON:
       {
           "tool": "tool_name",
           "parameters": {"param1": "value1"}
       }

    4. Markdown JSON:
       ```json
       {
           "tool": "tool_name",
           "parameters": {"param1": "value1"}
       }
       ```
    """

    def __init__(self, default_format: ToolCallFormat = ToolCallFormat.AUTO):
        """
        Initialize tool call parser.

        Args:
            default_format: Default format to use (AUTO = auto-detect)
        """
        self.default_format = default_format

        logger.info(f"ToolCallParser initialized: format={default_format.value}")

    def parse(
        self, text: str, format: Optional[ToolCallFormat] = None
    ) -> List[ParsedToolCall]:
        """
        Parse tool calls from text.

        Args:
            text: Text to parse
            format: Format to use (None = use default)

        Returns:
            List of ParsedToolCall objects
        """
        format = format or self.default_format

        if format == ToolCallFormat.AUTO:
            return self._parse_auto(text)
        elif format == ToolCallFormat.OPENAI_FUNCTION:
            return self._parse_openai_function(text)
        elif format == ToolCallFormat.XML:
            return self._parse_xml(text)
        elif format == ToolCallFormat.JSON:
            return self._parse_json(text)
        elif format == ToolCallFormat.MARKDOWN_JSON:
            return self._parse_markdown_json(text)
        else:
            logger.warning(f"Unknown format: {format}")
            return []

    def _parse_auto(self, text: str) -> List[ParsedToolCall]:
        """
        Auto-detect format and parse.

        Tries formats in order of likelihood based on heuristics.

        Args:
            text: Text to parse

        Returns:
            List of ParsedToolCall objects
        """
        # Try each format and return first that succeeds
        parsers = [
            (self._parse_markdown_json, "markdown_json"),
            (self._parse_xml, "xml"),
            (self._parse_openai_function, "openai_function"),
            (self._parse_json, "json"),
        ]

        for parser_func, format_name in parsers:
            try:
                results = parser_func(text)
                if results:
                    logger.debug(f"Auto-detected format: {format_name}")
                    return results
            except Exception as e:
                logger.debug(f"Format {format_name} failed: {e}")
                continue

        logger.warning("No tool calls found in text (auto-detect)")
        return []

    def _parse_openai_function(self, text: str) -> List[ParsedToolCall]:
        """
        Parse OpenAI function calling format.

        Format:
        {
            "name": "tool_name",
            "arguments": "{\"param1\": \"value1\"}"
        }

        Args:
            text: Text to parse

        Returns:
            List of ParsedToolCall objects
        """
        results = []

        try:
            data = json.loads(text)

            if isinstance(data, dict) and "name" in data:
                # Single function call
                name = data["name"]
                arguments = data.get("arguments", "{}")

                # Parse arguments (may be string or dict)
                if isinstance(arguments, str):
                    parameters = json.loads(arguments)
                else:
                    parameters = arguments

                results.append(
                    ParsedToolCall(
                        name=name,
                        parameters=parameters,
                        raw_text=text,
                        format=ToolCallFormat.OPENAI_FUNCTION,
                    )
                )

            elif isinstance(data, list):
                # Multiple function calls
                for item in data:
                    if "name" in item:
                        name = item["name"]
                        arguments = item.get("arguments", "{}")

                        if isinstance(arguments, str):
                            parameters = json.loads(arguments)
                        else:
                            parameters = arguments

                        results.append(
                            ParsedToolCall(
                                name=name,
                                parameters=parameters,
                                raw_text=json.dumps(item),
                                format=ToolCallFormat.OPENAI_FUNCTION,
                            )
                        )

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse as OpenAI function format: {e}")

        return results

    def _parse_xml(self, text: str) -> List[ParsedToolCall]:
        """
        Parse XML format.

        Format:
        <tool_call>
            <name>tool_name</name>
            <parameters>
                <param1>value1</param1>
            </parameters>
        </tool_call>

        Args:
            text: Text to parse

        Returns:
            List of ParsedToolCall objects
        """
        results = []

        # Find all <tool_call> blocks
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                # Wrap in root element for parsing
                xml_text = f"<tool_call>{match}</tool_call>"

                # Use defusedxml if available to prevent XXE attacks
                if DEFUSEDXML_AVAILABLE:
                    root = DefusedET.fromstring(xml_text)
                else:
                    # Fallback to standard parser with warning
                    # Note: This is vulnerable to XXE attacks
                    # Users should install defusedxml: pip install defusedxml
                    if not hasattr(self, "_xxe_warned"):
                        logger.warning(
                            "Using unsafe XML parser. Install defusedxml for security: "
                            "pip install defusedxml"
                        )
                        self._xxe_warned = True

                    # Basic safety: limit input size
                    if len(xml_text) > 10000:  # 10KB limit
                        logger.warning(f"XML input too large ({len(xml_text)} bytes), skipping")
                        continue

                    root = ET.fromstring(xml_text)

                # Extract name
                name_elem = root.find("name")
                if name_elem is None:
                    logger.warning("Missing <name> in tool_call")
                    continue

                name = name_elem.text

                # Extract parameters
                parameters = {}
                params_elem = root.find("parameters")

                if params_elem is not None:
                    for child in params_elem:
                        # Try to parse as JSON if possible
                        try:
                            parameters[child.tag] = json.loads(child.text)
                        except (json.JSONDecodeError, TypeError):
                            parameters[child.tag] = child.text

                results.append(
                    ParsedToolCall(
                        name=name,
                        parameters=parameters,
                        raw_text=xml_text,
                        format=ToolCallFormat.XML,
                    )
                )

            except ET.ParseError as e:
                logger.warning(f"Failed to parse XML tool call: {e}")
                continue

        return results

    def _parse_json(self, text: str) -> List[ParsedToolCall]:
        """
        Parse JSON format.

        Format:
        {
            "tool": "tool_name",
            "parameters": {"param1": "value1"}
        }

        Args:
            text: Text to parse

        Returns:
            List of ParsedToolCall objects
        """
        results = []

        try:
            data = json.loads(text)

            if isinstance(data, dict) and "tool" in data:
                # Single tool call
                name = data["tool"]
                parameters = data.get("parameters", {})

                results.append(
                    ParsedToolCall(
                        name=name,
                        parameters=parameters,
                        raw_text=text,
                        format=ToolCallFormat.JSON,
                    )
                )

            elif isinstance(data, list):
                # Multiple tool calls
                for item in data:
                    if isinstance(item, dict) and "tool" in item:
                        name = item["tool"]
                        parameters = item.get("parameters", {})

                        results.append(
                            ParsedToolCall(
                                name=name,
                                parameters=parameters,
                                raw_text=json.dumps(item),
                                format=ToolCallFormat.JSON,
                            )
                        )

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse as JSON format: {e}")

        return results

    def _parse_markdown_json(self, text: str) -> List[ParsedToolCall]:
        """
        Parse markdown JSON format.

        Format:
        ```json
        {
            "tool": "tool_name",
            "parameters": {"param1": "value1"}
        }
        ```

        Args:
            text: Text to parse

        Returns:
            List of ParsedToolCall objects
        """
        results = []

        # Find all ```json blocks
        pattern = r"```json\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            # Parse JSON content
            json_results = self._parse_json(match)
            results.extend(json_results)

        return results

    def extract_tool_calls_from_response(
        self, response_text: str
    ) -> List[ParsedToolCall]:
        """
        Extract tool calls from a full model response.

        This method handles responses that may contain both regular text
        and tool calls, extracting only the tool calls.

        Args:
            response_text: Full model response text

        Returns:
            List of ParsedToolCall objects
        """
        # Try to extract tool calls using auto-detection
        return self.parse(response_text, format=ToolCallFormat.AUTO)
