import json
import re
import logging

logger = logging.getLogger(__name__)

# Common refusal response patterns from LLMs
# These patterns indicate that the LLM is refusing to process the request
# rather than returning actual data. This list can be extended as needed.
LLM_REFUSAL_PATTERNS = [
    # English patterns
    "i am unable to",
    "i cannot process",
    "i cannot extract",
    "unable to process",
    "unable to extract",
    "cannot process",
    "cannot extract",
    "falls outside my",
    "not suitable for",
    "not appropriate",
    "i'm sorry",
    "i apologize",
    "i cannot help",
    "lacks clear",
    "does not contain",
    "fragmented excerpt",
    "incomplete sentences",
    "no entities",
    "no extractable",
    # Chinese patterns
    "无法处理",
    "不能处理",
    "无法提取",
    "不能提取",
    "不适合",
    "抱歉",
    "没有找到",
    "无法从"
]


def _fix_json_format_errors(json_str):
    """
    Attempt to fix common JSON format errors in LLM responses.
    
    This function tries to fix structural JSON errors while being careful
    not to modify string content. It uses a simple approach that may not
    catch all edge cases, but should handle most common LLM output errors.
    
    Args:
        json_str: The JSON string that may contain format errors
        
    Returns:
        Fixed JSON string
    """
    # Fix unclosed arrays: if we have an opening [ but no closing ]
    # Count brackets and add missing closing brackets
    # Note: This is a simple heuristic and may not work for all cases
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    # Fix unclosed objects: if we have an opening { but no closing }
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    # Fix missing comma before closing brace/bracket in arrays/objects
    # Use a more conservative pattern that tries to avoid matching inside strings
    # Pattern: "value"\n    "key" -> "value",\n    "key"
    # Only match if we're outside of strings (simple heuristic: even number of quotes before)
    # This is not perfect but should work for most cases
    def add_comma_between_quotes(match):
        before = json_str[:match.start()]
        # Count unescaped quotes before this position
        quote_count = len(re.findall(r'(?<!\\)"', before))
        # If even number of quotes, we're outside a string, safe to add comma
        if quote_count % 2 == 0:
            return '",\n    "'
        return match.group(0)
    
    json_str = re.sub(r'"\s*\n\s*"', add_comma_between_quotes, json_str)
    
    # Fix missing comma in arrays: "item"\n    ] -> "item",\n    ]
    def add_comma_before_bracket(match):
        before = json_str[:match.start()]
        quote_count = len(re.findall(r'(?<!\\)"', before))
        if quote_count % 2 == 0:
            return '",\n    ]'
        return match.group(0)
    
    json_str = re.sub(r'"\s*\n\s*\]', add_comma_before_bracket, json_str)
    
    # Fix missing comma in objects: "value"\n    } -> "value",\n    }
    def add_comma_before_brace(match):
        before = json_str[:match.start()]
        quote_count = len(re.findall(r'(?<!\\)"', before))
        if quote_count % 2 == 0:
            return '",\n    }'
        return match.group(0)
    
    json_str = re.sub(r'"\s*\n\s*\}', add_comma_before_brace, json_str)
    
    return json_str


def load_knowIE_data(respond, lang="en"):
    # Early check: if it's a refusal response without JSON, return empty dict
    if _is_refusal_response(respond):
        json_match = _extract_json_from_response(respond)
        if not json_match:
            logger.debug(f"LLM refusal response without JSON: {respond[:200]}...")
            return {}
    
    # Clean the response
    respond = _clean_json_response(respond)
    
    try:
        extract_ret = json.loads(respond)
    except (json.JSONDecodeError, ValueError) as e:
        extract_ret_str = modify_knowledge_unit(respond)
        try:
            left_pos = respond.find("{") if respond.find("{") >= 0 else 0
            right_pos = respond.rfind("}") + 1
            extract_ret_str = extract_ret_str[left_pos:right_pos].strip()
            extract_ret_str = extract_ret_str.replace("\\'", "'")
            extract_ret = json.loads(extract_ret_str)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            try:
                extract_ret_str = "{" + "{".join(respond.split("{")[1:])
                extract_ret_str = extract_ret_str.split("}\n}")[0] + "}\n}"
                pattern = r'(?<="Content": ")(.*?)(?=",\n    "Knowledge Type")'
                extract_ret_str = re.sub(
                    pattern,
                    lambda match: match.group(1).replace('"', r"\""),
                    extract_ret_str,
                )
                extract_ret = json.loads(extract_ret_str)
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                try:
                    if "```json" in extract_ret_str:
                        extract_ret_str = extract_ret_str.split("```json")[1]
                    if "output:\n" in extract_ret_str:
                        extract_ret_str = extract_ret_str.split("output:\n")[1]
                    if "```" in extract_ret_str:
                        extract_ret = extract_ret_str.split("```")[0]
                    else:
                        extract_ret = extract_ret_str.replace("\\'", "'").replace(
                            '\\"', '"'
                        )  # .replace('\"Sleazy\"' ,'\\\"Sleazy\\\"').replace('\\\"Planet of the Apes\\\"' ,'Planet of the Apes') #.replace("\\", '\\\\').replace("\'", "\\\'")
                    extract_ret = json.loads(extract_ret)
                except Exception as e:
                    logger.warning(
                        f"load_knowIE_data retry2 has exception {e} {respond[:500]}...",
                        exc_info=False,
                    )
                    try:
                        # Try to fix common JSON format errors
                        fixed_json = _fix_json_format_errors(extract_ret_str)
                        extract_ret = json.loads(fixed_json)
                        logger.debug("Successfully fixed JSON format errors")
                    except (json.JSONDecodeError, ValueError) as e:
                        try:
                            extract_ret = json.loads(extract_ret_str + "}")
                        except (json.JSONDecodeError, ValueError) as e:
                            # Final check: if it's a refusal response, return empty dict
                            if _is_refusal_response(respond):
                                logger.debug(f"LLM refusal response, returning empty dict: {respond[:200]}...")
                                return {}
                            raise ValueError(
                                "the output KnowUnit str is invalid: " + respond[:500]
                            )
    return extract_ret


def _fix_missing_quotes_in_property_names(json_str):
    """
    Fix missing quotes around property names in JSON.
    Pattern: property_name": -> "property_name":
    
    Args:
        json_str: JSON string that may have missing quotes
        
    Returns:
        Fixed JSON string
    """
    # Fix common JSON format errors: missing quotes around property names
    # Pattern: property_name": -> "property_name":
    # Match property names that are missing opening quote (e.g., Name": -> "Name":)
    return re.sub(r'(\s+)([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1"\2":', json_str)


def _clean_json_response(respond):
    """
    Clean common formatting issues in LLM JSON responses.
    
    Args:
        respond: Raw response string from LLM
        
    Returns:
        Cleaned response string
    """
    if not isinstance(respond, str):
        respond = str(respond)
    
    # Fix repeated "json" strings (e.g., ```jsonjsonjson...)
    respond = re.sub(r'```json+', '```json', respond)
    respond = re.sub(r'json+```', 'json```', respond)
    
    # Remove markdown code blocks
    if "```json" in respond:
        respond = respond.split("```json")[1]
    if "```" in respond:
        respond = respond.split("```")[0]
    if "output:" in respond:
        respond = respond.split("output:")[1]
    if "output:\n" in respond:
        respond = respond.split("output:\n")[1]
    
    return respond.strip()


def _is_refusal_response(respond):
    """
    Check if the response is a refusal from LLM.
    
    Args:
        respond: The response string to check
        
    Returns:
        True if it's a refusal response, False otherwise
    """
    if not isinstance(respond, str):
        respond = str(respond)
    
    respond_lower = respond.lower()
    return any(pattern in respond_lower for pattern in LLM_REFUSAL_PATTERNS)


def _extract_json_from_response(respond):
    """
    Try to extract JSON from a response that may contain explanatory text.
    
    Args:
        respond: Response string that may contain JSON
        
    Returns:
        Extracted JSON string, or None if no JSON found
    """
    # Look for JSON array or object
    json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', respond)
    if json_match:
        return json_match.group(1)
    return None


def _fix_unterminated_string(json_str):
    """Try to fix unterminated strings in JSON by finding the last complete object."""
    if not json_str.strip().startswith('['):
        return None
    
    # Simple strategy: find the last "}" that appears to close a complete object
    # We'll look for patterns that indicate the end of an object in an array
    # Pattern: "}\n" or "},\n" followed by whitespace or end of string
    
    # Try to find all potential object endings
    # Look for "}" that might close an object, checking if it's followed by comma/newline
    potential_ends = []
    in_string = False
    escape_next = False
    
    for i, char in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '}':
            # Check what comes after this brace
            after = json_str[i+1:].strip()
            # If it's followed by comma, newline, or end of string, it might be a complete object
            if not after or after.startswith(',') or after.startswith('\n') or after.startswith(']'):
                potential_ends.append(i)
    
    # Try each potential end position, starting from the last one
    for end_pos in reversed(potential_ends):
        try:
            # Extract up to this position and try to parse
            test_str = json_str[:end_pos + 1].rstrip().rstrip(',')
            if test_str.strip().startswith('[') and not test_str.rstrip().endswith(']'):
                test_str = test_str + '\n]'
            # Try to parse it
            json.loads(test_str)
            # If parsing succeeds, this is a valid cutoff point
            return test_str
        except (json.JSONDecodeError, ValueError, IndexError):
            continue
    
    return None


def load_NER_data(respond):
    # Early detection: check for refusal responses from LLM
    if _is_refusal_response(respond):
        # Check if there's any JSON in the response despite the refusal
        json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', respond)
        if not json_match:
            # No JSON found, return empty list
            logger.debug(f"LLM returned refusal response without JSON, returning empty list: {respond[:200]}...")
            return []
        # If JSON is found, continue processing
    
    try:
        extract_ret_str = respond.replace(
            'Before" trilogy,', 'Before trilogy",'
        ).replace('I"s', "I's")
        if "```json" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("```json")[1]
        if "output:" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("output:")[1]
        if "```" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("```")[0]
        extract_ret_str = extract_ret_str.replace("\\'", "'")
        
        # If response doesn't start with [ or {, try to extract JSON from it
        extract_ret_str_stripped = extract_ret_str.strip()
        if not (extract_ret_str_stripped.startswith('[') or extract_ret_str_stripped.startswith('{')):
            # Look for JSON array or object in the response
            json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', extract_ret_str_stripped)
            if json_match:
                # Found JSON in the response, extract and use it
                extract_ret_str = json_match.group(1)
            else:
                # No JSON found and response doesn't start with JSON, check if it's a refusal
                if _is_refusal_response(respond):
                    logger.debug(f"LLM returned refusal response without JSON, returning empty list: {respond[:200]}...")
                    return []
        
        extract_ret = json.loads(extract_ret_str)
    except json.JSONDecodeError as e:
        # Check if it's an unterminated string error
        if "Unterminated string" in str(e) or "Unterminated" in str(e):
            # Try to fix by finding the last complete object
            fixed_str = _fix_unterminated_string(extract_ret_str)
            if fixed_str:
                try:
                    extract_ret = json.loads(fixed_str)
                    return extract_ret  # Successfully fixed, return immediately
                except (json.JSONDecodeError, ValueError) as e:
                    # If fixing didn't work, continue to other strategies
                    pass
        
        # If we haven't succeeded yet, try other strategies
        extract_ret = None
        try:
            extract_ret_str = "[" + "[".join(extract_ret_str.split("[")[1:])
            extract_ret_str = "]".join(extract_ret_str.split("]")[:-1]) + "]"
            # Fix common JSON format errors: missing quotes around property names
            # Pattern: property_name": -> "property_name":
            # Match property names that are missing opening quote (e.g., Name": -> "Name":)
            extract_ret_str = _fix_missing_quotes_in_property_names(extract_ret_str)
            extract_ret = json.loads(extract_ret_str)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            try:
                # Try fixing unterminated string again after other fixes
                fixed_str = _fix_unterminated_string(extract_ret_str)
                if fixed_str:
                    extract_ret_str = fixed_str
                else:
                    extract_ret_str = extract_ret_str.strip() + "}]"
                # Fix common JSON format errors: missing quotes around property names
                extract_ret_str = _fix_missing_quotes_in_property_names(extract_ret_str)
                extract_ret = json.loads(extract_ret_str)
            except (json.JSONDecodeError, ValueError) as e2:
                # Before raising error, check if it's a refusal response
                if _is_refusal_response(respond):
                    logger.debug(f"LLM returned refusal response, returning empty list: {respond[:200]}...")
                    return []
                raise ValueError("the output NER str is invalid: " + respond[:1000])
    
    if extract_ret is None:
        # If we still haven't parsed successfully, try one more time with general exception handling
        try:
            extract_ret_str = "[" + "[".join(extract_ret_str.split("[")[1:])
            extract_ret_str = "]".join(extract_ret_str.split("]")[:-1]) + "]"
            extract_ret_str = _fix_missing_quotes_in_property_names(extract_ret_str)
            extract_ret = json.loads(extract_ret_str)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            try:
                # Try fixing unterminated string
                fixed_str = _fix_unterminated_string(extract_ret_str)
                if fixed_str:
                    extract_ret_str = fixed_str
                else:
                    extract_ret_str = extract_ret_str.strip() + "}]"
                extract_ret_str = _fix_missing_quotes_in_property_names(extract_ret_str)
                extract_ret = json.loads(extract_ret_str)
            except (json.JSONDecodeError, ValueError) as e2:
                # Before raising error, check if it's a refusal response
                if _is_refusal_response(respond):
                    logger.debug(f"LLM returned refusal response, returning empty list: {respond[:200]}...")
                    return []
                raise ValueError("the output NER str is invalid: " + respond[:1000])
    return extract_ret


def load_SPO_data(respond):
    extract_ret_str = respond
    try:
        if "output:\n" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("output:\n")[1]
        if "```json\n" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("```json\n")[1]
        if "\n```" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("\n```")[0]
        if "input:\n" in extract_ret_str:
            extract_ret_str = extract_ret_str.split("input:\n")[0]
        extract_ret_str = extract_ret_str.replace("\\'", "'")
        rst = check_data(extract_ret_str, "spo", "zh")
        if rst and len(rst) > 0:
            return rst
        else:
            return []
            # raise ValueError("the output SPO str is invalid: " + respond)

    except (json.JSONDecodeError, ValueError, IndexError, AttributeError) as e:
        matches = re.findall(r"\[([^[\[\]].*)\]", extract_ret_str)
        unique2quadruple = {}

        for ele in matches:
            ele = (
                ele.strip()
                .strip("[")
                .strip("]")
                .replace(" ", "")
                .strip('"')
                .strip('",')
                .strip('"')
            )
            quadruple = [
                sub.strip('"').strip(",").strip("'") for sub in ele.split('","')
            ]
            if len(quadruple) == 4:
                unique2quadruple["-".join(quadruple)] = quadruple
            elif len(quadruple) == 3:
                unique2quadruple["-".join(quadruple)] = quadruple + [""]

            rst = list(unique2quadruple.values())
            if len(rst) > 0:
                return rst
            else:
                for quanple in extract_ret_str.split("]"):
                    try:
                        content = quanple.split("[")[1]
                        quadruple = [
                            ele.strip().strip('"').strip("'")
                            for ele in content.replace("\", '", '", "')
                            .replace("', \"", '", "')
                            .replace("', '", '", "')
                            .split('", "')
                        ]
                        if len(quadruple) == 4:
                            unique2quadruple["-".join(quadruple)] = quadruple
                        elif len(quadruple) == 3:
                            unique2quadruple["-".join(quadruple)] = quadruple + [""]
                        elif len(quadruple) == 5:
                            unique2quadruple["-".join(quadruple)] = [
                                quadruple[0],
                                quadruple[1],
                                quadruple[2] + " " + quadruple[3],
                                quadruple[4],
                            ]
                    except (ValueError, IndexError, AttributeError) as e:
                        continue
                rst = list(unique2quadruple.values())
                if len(rst) > 0:
                    return rst
                else:
                    return []
                    # raise ValueError("the output SPO str is invalid: " + respond)


def modify_knowledge_unit(text, lang="zh"):
    # 定义正则表达式模式
    if lang == "zh":
        pattern = r'"知识点\d+名称"\s*:\s*"([^"]+)"\s*,'
    else:
        pattern = r'"knowledge unit \d+ Name"\s*:\s*"([^"]+)",'
    modified_text = re.sub(pattern, r'"\1":', text)
    return modified_text


def check_data(line, data_type="knowIE", language="zh"):
    """
    Check and parse data from LLM response.
    
    Args:
        line: The response string from LLM
        data_type: Type of data expected ("knowIE", "ner", "spo")
        language: Language of the response ("zh" or "en")
    
    Returns:
        Parsed data structure or None if parsing fails
    """
    if not line or not isinstance(line, str):
        return None
    
    line_stripped = line.strip()
    if not line_stripped:
        return None
    
    # Early check: if it's a refusal response without JSON, return None
    if _is_refusal_response(line):
        json_match = _extract_json_from_response(line_stripped)
        if not json_match:
            logger.debug(f"LLM refusal response without JSON: {line[:200]}...")
            return None
        # If JSON found, use it
        line_stripped = json_match
        line = json_match
    # Early detection: if response doesn't start with JSON, try to extract JSON from it
    elif not (line_stripped.startswith('[') or line_stripped.startswith('{')):
        json_match = _extract_json_from_response(line_stripped)
        if json_match:
            logger.debug(f"Extracted JSON from explanation text: {json_match[:100]}...")
            line_stripped = json_match
            line = json_match
        else:
            # No JSON found, return None
            logger.debug(f"No JSON found in response, returning None")
            return None
    
    try:
        info = json.loads(line)
    except Exception as e:
        logger.warning(f"check_data retry0 has exception {e} {line[:200]}...", exc_info=False)
        try:
            info = json.loads(line.replace("```json", "").replace("\n``", ""))
        except Exception as e:
            logger.warning(f"check_data retry1 has exception {e} {line[:200]}...", exc_info=False)
            try:
                info = json.loads(
                    line.replace("```json", "")
                    .replace("\n``", "")
                    .replace("\\", "\\\\")
                )
            except Exception as e:
                logger.warning(
                    f"check_data retry2 has exception {e} {line[:200]}...", exc_info=False
                )
                return None
    if data_type == "knowIE":
        if not isinstance(info, dict):
            return None
        check_data = {}
        for name in info:

            if (
                language == "zh"
                and "知识点" not in name
                and len(
                    set(info[name].keys())
                    & set(
                        [
                            "内容",
                            "知识类型",
                            "结构化内容",
                            "领域本体",
                            "核心实体",
                            "关联问",
                            "扩展知识点",
                        ]
                    )
                )
                >= 6
            ):
                check_data[name] = info[name]
            # elif language == "en" and "knowledge unit" not in name and lsn(set(info[name].keys()) & set([ "Name","Type", "Domain Ontology", "Description","Standard Name", "Synonyms"])) >=6:
            elif language == "en" and isinstance(info[name], dict):
                if (
                    "knowledge unit" not in name
                    and len(
                        set(info[name].keys())
                        & set(
                            [
                                "Content",
                                "Knowledge Type",
                                "Structured Content",
                                "Domain Ontology",
                                "Core Entities",
                                "Related Query",
                                "Extended Knowledge Points",
                            ]
                        )
                    )
                    >= 6
                ):
                    check_data[name] = info[name]
        if len(check_data) > 0:
            return check_data

    elif data_type == "ner":
        check_data = []
        if not isinstance(info, list):
            return None

        for ner in info:
            if language == "en" and isinstance(ner, dict):
                if (
                    len(
                        set(ner.keys())
                        & set(
                            [
                                "Name",
                                "Type",
                                "Domain Ontology",
                                "Description",
                                "Standard Name",
                                "Synonyms",
                            ]
                        )
                    )
                    == 6
                ):
                    check_data.append(ner)
            if language == "zh" and isinstance(ner, dict):
                if (
                    len(set(ner.keys()) & set(["名称", "类型", "领域本体", "解释", "标准名", "同义词"]))
                    == 6
                ):
                    check_data.append(ner)
        if len(check_data) > 0:
            return check_data

    elif data_type == "spo":
        if not isinstance(info, list):
            return None
        check_data = []
        # print(info)
        valid = {}
        for spo in info:
            try:
                if isinstance(spo, list) and (len(spo) < 3 or len(spo[1].strip()) == 0):
                    continue
                elif isinstance(spo, list) and len(spo) == 4:
                    if spo[0] == spo[3]:
                        spo[3] = ""
                    valid["_".join(spo)] = spo
                elif isinstance(spo, list) and len(spo) == 3:
                    spo = spo + [""]
                    valid["_".join(spo)] = spo
            except Exception as e:
                logger.warning(
                    f"check_data spo parsed. has exception {e} {spo}", exc_info=True
                )
                continue
        check_data = list(valid.values())
        if len(check_data) > 0:
            return check_data
    return None
