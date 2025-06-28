import os
import json
import shutil


def extract_only_numbers(data):
    """
    Recursively extract only numeric values from any JSON structure
    """
    numbers = []

    def extract_recursive(item):
        if isinstance(item, (int, float)):
            # Only add finite numbers (not NaN or infinity)
            if (
                str(item).lower() not in ["nan", "inf", "-inf"] and item == item
            ):  # NaN != NaN
                numbers.append(item)
        elif isinstance(item, list):
            for element in item:
                extract_recursive(element)
        elif isinstance(item, dict):
            for value in item.values():
                extract_recursive(value)
        elif isinstance(item, str):
            # Try to convert string to number
            try:
                num = float(item)
                if str(num).lower() not in ["nan", "inf", "-inf"] and num == num:
                    numbers.append(num)
            except (ValueError, TypeError):
                pass  # Skip non-numeric strings

    extract_recursive(data)
    return numbers


def clean_json_file(filepath, backup=True):
    """
    Clean a single JSON file to contain only numbers
    """
    try:
        # Read the original file
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return "parse_error", f"Could not parse JSON: {str(e)}"

        # Extract only numbers
        numbers = extract_only_numbers(data)

        if not numbers:
            return "no_numbers", "No valid numbers found in file"

        # Create backup if requested
        if backup:
            backup_path = filepath + ".backup"
            if not os.path.exists(backup_path):  # Don't overwrite existing backups
                shutil.copy2(filepath, backup_path)

        # Create clean JSON with only numbers
        clean_json = json.dumps(numbers, indent=4)

        # Check if anything changed
        original_count = count_elements(data)
        if len(numbers) == original_count and isinstance(data, list):
            # Check if all elements were already numbers
            if all(isinstance(x, (int, float)) for x in data):
                return "already_clean", None

        # Write the cleaned content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(clean_json)

        return (
            "cleaned",
            f"Extracted {len(numbers)} numbers from {original_count} elements",
        )

    except Exception as e:
        return "error", str(e)


def count_elements(data):
    """
    Count total elements in a nested structure
    """
    count = 0

    def count_recursive(item):
        nonlocal count
        if isinstance(item, list):
            for element in item:
                count += 1
                if isinstance(element, (list, dict)):
                    count_recursive(element)
        elif isinstance(item, dict):
            for value in item.values():
                count += 1
                if isinstance(value, (list, dict)):
                    count_recursive(value)
        else:
            count += 1

    if isinstance(data, (list, dict)):
        count_recursive(data)
    else:
        count = 1

    return count


def clean_json_files_in_directory(root_directory, target_suffixes=None, backup=True):
    """
    Clean all JSON files in directory structure
    """
    if target_suffixes is None:
        target_suffixes = ["diff_end_all", "diff_start_all"]

    results = {
        "already_clean": [],
        "cleaned": [],
        "no_numbers": [],
        "parse_error": [],
        "error": [],
    }

    stats = {
        "total_files": 0,
        "total_numbers_extracted": 0,
        "total_original_elements": 0,
    }

    # Walk through all directories
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".json"):
                # Check if file matches target suffixes (if specified)
                if target_suffixes and not any(
                    suffix in file for suffix in target_suffixes
                ):
                    continue

                filepath = os.path.join(root, file)
                stats["total_files"] += 1

                status, message = clean_json_file(filepath, backup)
                results[status].append(filepath)

                if status == "already_clean":
                    print(f"✓ Already clean: {filepath}")
                elif status == "cleaned":
                    print(f"🧹 Cleaned: {filepath} - {message}")
                    # Extract numbers for stats
                    if message:
                        try:
                            parts = message.split()
                            if len(parts) >= 4:
                                numbers_count = int(parts[1])
                                original_count = int(parts[4])
                                stats["total_numbers_extracted"] += numbers_count
                                stats["total_original_elements"] += original_count
                        except Exception as e:
                            print(e)
                elif status == "no_numbers":
                    print(f"⚠️ No numbers found: {filepath}")
                elif status == "parse_error":
                    print(f"❌ Parse error: {filepath} - {message}")
                elif status == "error":
                    print(f"⚠️ Error: {filepath} - {message}")

    return results, stats


def main():
    # Configuration
    root_directory = "./data_cache/outputs"  # Change this to your data directory
    target_suffixes = [
        "diff_end_all",
        "diff_start_all",
    ]  # Only process files with these suffixes
    # Set target_suffixes = None to process ALL JSON files
    create_backup = True  # Set to False if you don't want backup files

    print("JSON Number Cleaner")
    print("=" * 60)
    print(f"Root directory: {os.path.abspath(root_directory)}")
    print(
        f"Target suffixes: {target_suffixes if target_suffixes else 'ALL JSON files'}"
    )
    print(f"Create backups: {create_backup}")
    print()

    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist!")
        return

    # Clean all JSON files
    results, stats = clean_json_files_in_directory(
        root_directory, target_suffixes, create_backup
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Already clean: {len(results['already_clean'])}")
    print(f"Successfully cleaned: {len(results['cleaned'])}")
    print(f"No numbers found: {len(results['no_numbers'])}")
    print(f"Parse errors: {len(results['parse_error'])}")
    print(f"Other errors: {len(results['error'])}")

    if stats["total_numbers_extracted"] > 0:
        print("\nData processed:")
        print(f"Total numbers extracted: {stats['total_numbers_extracted']:,}")
        print(f"Total original elements: {stats['total_original_elements']:,}")
        reduction_percent = (
            (stats["total_original_elements"] - stats["total_numbers_extracted"])
            / stats["total_original_elements"]
        ) * 100
        print(f"Data reduction: {reduction_percent:.1f}%")

    # Show problematic files
    if results["no_numbers"]:
        print("\nFiles with no numbers found:")
        for filepath in results["no_numbers"][:10]:  # Show first 10
            print(f"  - {filepath}")
        if len(results["no_numbers"]) > 10:
            print(f"  ... and {len(results['no_numbers']) - 10} more")

    if results["parse_error"]:
        print("\nFiles with parse errors:")
        for filepath in results["parse_error"][:10]:  # Show first 10
            print(f"  - {filepath}")
        if len(results["parse_error"]) > 10:
            print(f"  ... and {len(results['parse_error']) - 10} more")

    if results["error"]:
        print("\nFiles with other errors:")
        for filepath in results["error"][:10]:  # Show first 10
            print(f"  - {filepath}")
        if len(results["error"]) > 10:
            print(f"  ... and {len(results['error']) - 10} more")

    if create_backup:
        print("\nBackup files created with .backup extension")
        print("You can delete them after verifying the results")


if __name__ == "__main__":
    main()
