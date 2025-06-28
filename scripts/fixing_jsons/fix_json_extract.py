import os
import json
import re
import shutil


def fix_malformed_json(content):
    """
    Fix common JSON formatting issues
    """
    # Remove any trailing commas before closing brackets
    content = re.sub(r",\s*]", "]", content)
    content = re.sub(r",\s*}", "}", content)

    # Fix the specific issue: numbers followed by nested empty arrays
    # Pattern: number,\n[\n]]
    content = re.sub(r"(\d+\.?\d*(?:[eE][+-]?\d+)?),?\s*\[\s*\]\s*\]", r"\1]", content)

    # Remove standalone empty arrays that appear after commas
    content = re.sub(r",\s*\[\s*\]", "", content)

    # Fix missing commas between numbers on separate lines
    # Pattern: number followed by whitespace/newline then another number
    content = re.sub(
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\n\s*(\d)", r"\1,\n    \2", content
    )

    # Fix missing commas between numbers with minimal spacing
    content = re.sub(r"(\d+\.?\d*(?:[eE][+-]?\d+)?)\s+(\d)", r"\1, \2", content)

    # Handle cases where there might be missing opening/closing brackets
    content = content.strip()
    if not content.startswith("[") and not content.startswith("{"):
        # If it looks like a list of numbers, wrap in brackets
        if re.match(r"^\s*-?\d+\.?\d*", content):
            content = "[" + content + "]"

    # Clean up multiple consecutive closing brackets
    content = re.sub(r"\]\s*\]\s*\]", "]]", content)
    content = re.sub(r"\]\s*\]", "]", content)

    return content


def flatten_nested_arrays(data):
    """
    Flatten nested arrays and remove empty arrays
    """

    def flatten_recursive(item):
        if isinstance(item, list):
            result = []
            for element in item:
                if isinstance(element, list):
                    result.extend(flatten_recursive(element))
                elif element is not None:  # Skip None values
                    result.append(element)
            return result
        else:
            return [item] if item is not None else []

    return flatten_recursive(data)


def extract_numbers_as_json(content):
    """
    Extract all numbers from content and format as JSON array
    """
    # Find all floating point and integer numbers
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", content)

    # Convert to floats where possible, keep as strings otherwise
    processed_numbers = []
    for num in numbers:
        try:
            if "." in num or "e" in num.lower():
                processed_numbers.append(float(num))
            else:
                processed_numbers.append(int(num))
        except ValueError:
            continue

    return json.dumps(processed_numbers, indent=4)


def fix_json_file(filepath, backup=True):
    """
    Fix a single JSON file
    """
    try:
        # Read the original file
        with open(filepath, "r", encoding="utf-8") as f:
            original_content = f.read()

        # First, try to parse as-is
        try:
            json.loads(original_content)
            return "already_valid", None
        except json.JSONDecodeError as e:
            original_error = str(e)

        # Create backup if requested
        if backup:
            backup_path = filepath + ".backup"
            shutil.copy2(filepath, backup_path)

        # Try to fix the content
        fixed_content = fix_malformed_json(original_content)

        # Test if the fix worked
        try:
            parsed_data = json.loads(fixed_content)

            # If parsed successfully, check if we need to flatten nested arrays
            if isinstance(parsed_data, list):
                flattened_data = flatten_nested_arrays(parsed_data)
                # Only keep numeric values
                numeric_data = [
                    x for x in flattened_data if isinstance(x, (int, float))
                ]

                if len(numeric_data) != len(flattened_data) or len(
                    flattened_data
                ) != len(parsed_data):
                    # Data was modified, save the cleaned version
                    cleaned_json = json.dumps(numeric_data, indent=4)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(cleaned_json)
                    return "cleaned_and_fixed", None
                else:
                    # Data was the same, just save the fixed formatting
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(fixed_content)
                    return "fixed", None
            else:
                # Not a list, just save the fixed content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                return "fixed", None

        except json.JSONDecodeError:
            # If fixing didn't work, try extracting numbers
            try:
                extracted_json = extract_numbers_as_json(original_content)
                json.loads(extracted_json)  # Verify it's valid

                # Write the extracted content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(extracted_json)
                return "extracted", None
            except Exception as e:
                print(e)
                # Restore original file if backup exists
                if backup and os.path.exists(backup_path):
                    shutil.copy2(backup_path, filepath)
                return "failed", original_error

    except Exception as e:
        return "error", str(e)


def fix_json_files_in_directory(root_directory, target_suffixes=None, backup=True):
    """
    Fix all JSON files in directory structure
    """
    if target_suffixes is None:
        target_suffixes = ["diff_end_all", "diff_start_all"]

    results = {
        "already_valid": [],
        "fixed": [],
        "cleaned_and_fixed": [],
        "extracted": [],
        "failed": [],
        "error": [],
    }

    # Walk through all directories
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".json"):
                # Check if file matches target suffixes
                if target_suffixes and not any(
                    suffix in file for suffix in target_suffixes
                ):
                    continue

                filepath = os.path.join(root, file)
                status, error = fix_json_file(filepath, backup)
                results[status].append(filepath)

                if status == "already_valid":
                    print(f"✓ Already valid: {filepath}")
                elif status == "fixed":
                    print(f"🔧 Fixed: {filepath}")
                elif status == "cleaned_and_fixed":
                    print(f"🧹 Cleaned and fixed: {filepath}")
                elif status == "extracted":
                    print(f"📊 Extracted numbers: {filepath}")
                elif status == "failed":
                    print(f"❌ Failed to fix: {filepath} - {error}")
                elif status == "error":
                    print(f"⚠️ Error processing: {filepath} - {error}")

    return results


def main():
    # Configuration
    root_directory = "./data_cache/outputs"  # Change this to your data directory
    target_suffixes = [
        "diff_end_all",
        "diff_start_all",
    ]  # Only fix files with these suffixes
    create_backup = True  # Set to False if you don't want backup files

    print("JSON File Fixer")
    print("=" * 50)
    print(f"Root directory: {os.path.abspath(root_directory)}")
    print(f"Target suffixes: {target_suffixes}")
    print(f"Create backups: {create_backup}")
    print()

    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist!")
        return

    # Fix all JSON files
    results = fix_json_files_in_directory(
        root_directory, target_suffixes, create_backup
    )

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Already valid: {len(results['already_valid'])}")
    print(f"Successfully fixed: {len(results['fixed'])}")
    print(f"Cleaned and fixed: {len(results['cleaned_and_fixed'])}")
    print(f"Numbers extracted: {len(results['extracted'])}")
    print(f"Failed to fix: {len(results['failed'])}")
    print(f"Errors: {len(results['error'])}")
    print(f"Total processed: {sum(len(v) for v in results.values())}")

    if results["failed"]:
        print("\nFailed files:")
        for filepath in results["failed"]:
            print(f"  - {filepath}")

    if results["error"]:
        print("\nError files:")
        for filepath in results["error"]:
            print(f"  - {filepath}")

    if create_backup:
        print("\nBackup files created with .backup extension")
        print("You can delete them after verifying the fixes worked correctly")


if __name__ == "__main__":
    main()
