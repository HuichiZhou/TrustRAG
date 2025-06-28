#!/usr/bin/env python3
"""
TrustRAG Unneeded Logs Cleaner
Removes logs that meet certain criteria for being unnecessary
"""

from pathlib import Path
from collections import defaultdict


class TrustRAGLogsCleaner:
    def __init__(self, logs_directory, dry_run=True):
        self.logs_dir = Path(logs_directory)
        self.dry_run = dry_run
        self.deleted_files = []
        self.processed_folders = []

    def get_experiment_base_name(self, filename):
        """Extract the base experiment name without the both_sides/both_combined suffix"""
        # Remove .log extension
        name = filename.replace(".log", "")

        # Remove the both_sides or both_combined suffix
        if name.endswith("_no_questions_both_sides"):
            base_name = name.replace("_no_questions_both_sides", "")
        elif name.endswith("_both_combined"):
            base_name = name.replace("_both_combined", "")
        else:
            # If it doesn't end with either, return the full name
            base_name = name

        return base_name

    def should_remove_file(self, filename):
        """Determine if a file should be removed based on criteria"""
        removal_reasons = []

        # Criterion 1: attack_none with q1 or q5 (should only keep q3)
        if "attack_none" in filename and (
            "adv_per_query1" in filename or "adv_per_query5" in filename
        ):
            removal_reasons.append("attack_none_wrong_query_count")

        # Add more criteria here as needed
        # Example:
        # if "some_other_criterion" in filename:
        #     removal_reasons.append("some_other_reason")

        return removal_reasons

    def find_files_to_remove(self, folder_path):
        """Find all files that should be removed based on criteria"""
        log_files = list(folder_path.glob("*.log"))

        # Group files by their base experiment name for duplicate detection
        experiment_groups = defaultdict(list)

        for log_file in log_files:
            base_name = self.get_experiment_base_name(log_file.name)
            experiment_groups[base_name].append(log_file)

        files_to_remove = []

        # 1. Find both_sides/both_combined duplicates
        for base_name, files in experiment_groups.items():
            if len(files) == 2:
                # Check if we have both versions
                both_sides_file = None
                both_combined_file = None

                for file in files:
                    if file.name.endswith("_no_questions_both_sides.log"):
                        both_sides_file = file
                    elif file.name.endswith("_both_combined.log"):
                        both_combined_file = file

                # If we have both, mark both_sides for deletion
                if both_sides_file and both_combined_file:
                    files_to_remove.append(
                        {
                            "reason": "both_sides_duplicate",
                            "base_name": base_name,
                            "to_delete": both_sides_file,
                            "to_keep": both_combined_file,
                        }
                    )

        # 2. Find files that meet removal criteria (regardless of duplicates)
        for log_file in log_files:
            removal_reasons = self.should_remove_file(log_file.name)

            for reason in removal_reasons:
                # Check if this file is not already marked for removal as a duplicate
                already_marked = any(
                    item["to_delete"] == log_file for item in files_to_remove
                )

                if not already_marked:
                    files_to_remove.append(
                        {
                            "reason": reason,
                            "base_name": log_file.name.replace(".log", ""),
                            "to_delete": log_file,
                            "to_keep": None,
                        }
                    )

        return files_to_remove

    def clean_folder(self, folder_path):
        """Clean files in a single folder based on removal criteria"""
        print(f"\n📁 Processing folder: {folder_path.name}")

        files_to_remove = self.find_files_to_remove(folder_path)

        if not files_to_remove:
            print("   ✅ No files to remove")
            return []

        deleted_in_folder = []

        for item in files_to_remove:
            file_to_delete = item["to_delete"]
            file_to_keep = item.get("to_keep")
            reason = item["reason"]

            if reason == "both_sides_duplicate":
                print("   🔍 Found both_sides/both_combined duplicate:")
                print(f"      🗑️  DELETE: {file_to_delete.name}")
                print(f"      ✅ KEEP:   {file_to_keep.name}")
            elif reason == "attack_none_wrong_query_count":
                print("   🎯 Found unneeded attack_none log:")
                print(f"      🗑️  DELETE: {file_to_delete.name}")
                print("      📝 REASON: attack_none should only use q3")
            else:
                print("   🗑️  Found unneeded log:")
                print(f"      🗑️  DELETE: {file_to_delete.name}")
                print(f"      📝 REASON: {reason}")

            if not self.dry_run:
                try:
                    file_to_delete.unlink()  # Delete the file
                    print(f"      ✅ Successfully deleted: {file_to_delete.name}")
                    deleted_in_folder.append(
                        {"file": str(file_to_delete), "reason": reason}
                    )
                except Exception as e:
                    print(f"      ❌ Error deleting {file_to_delete.name}: {e}")
            else:
                print(f"      📝 DRY RUN - Would delete: {file_to_delete.name}")
                deleted_in_folder.append(
                    {"file": str(file_to_delete), "reason": reason}
                )

        return deleted_in_folder

    def clean_all_folders(self):
        """Clean unneeded files in all experiment folders"""
        print("🧹 TrustRAG Unneeded Logs Cleaner")
        print("=" * 50)

        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will actually be deleted")
        else:
            print("⚠️  LIVE MODE - Files will be permanently deleted!")

        print("📋 Removal criteria:")
        print("   1. 'both_sides' when 'both_combined' exists (keep both_combined)")
        print("   2. 'attack_none' with q1 or q5 (keep only q3 for attack_none)")
        print("   3. [Add more criteria in should_remove_file() method]")

        # Find all experiment folders
        folders = [
            item
            for item in self.logs_dir.iterdir()
            if item.is_dir() and item.name.startswith("dataset_")
        ]

        print(f"\n📂 Found {len(folders)} experiment folders to process")

        total_deleted = []

        for folder in folders:
            deleted_in_folder = self.clean_folder(folder)
            total_deleted.extend(deleted_in_folder)
            self.processed_folders.append(folder.name)

        # Summary
        print("\n" + "=" * 50)
        print("📊 CLEANUP SUMMARY")
        print("=" * 50)

        if total_deleted:
            # Count by reason
            reason_counts = {}
            for item in total_deleted:
                reason = item["reason"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            print(
                f"🗑️  Total files {'marked for deletion' if self.dry_run else 'deleted'}: {len(total_deleted)}"
            )

            print("   Breakdown by reason:")
            for reason, count in reason_counts.items():
                emoji = "🔄" if reason == "both_sides_duplicate" else "🎯"
                print(f"   {emoji} {reason}: {count}")

            print("\n📝 List of deleted files:")
            for item in total_deleted:
                reason_emoji = (
                    "🔄" if item["reason"] == "both_sides_duplicate" else "🎯"
                )
                print(f"   {reason_emoji} {item['file']} ({item['reason']})")
        else:
            print("✅ No files found matching removal criteria!")

        print(f"\n📁 Processed {len(self.processed_folders)} folders")

        if self.dry_run:
            print("\n💡 To actually delete files, run with dry_run=False")
            print(
                "💡 To add more removal criteria, edit the should_remove_file() method"
            )

        self.deleted_files = total_deleted
        return total_deleted

    def generate_report(self, output_file="unneeded_logs_cleanup_report.txt"):
        """Generate a detailed report of the cleanup operation"""
        with open(output_file, "w") as f:
            f.write("TrustRAG Unneeded Logs Cleanup Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}\n")
            f.write(f"Processed folders: {len(self.processed_folders)}\n")
            f.write(f"Files deleted: {len(self.deleted_files)}\n\n")

            if self.deleted_files:
                # Group by reason
                reason_groups = {}
                for item in self.deleted_files:
                    reason = item["reason"]
                    if reason not in reason_groups:
                        reason_groups[reason] = []
                    reason_groups[reason].append(item)

                f.write("Deletion breakdown:\n")
                for reason, items in reason_groups.items():
                    f.write(f"  - {reason}: {len(items)}\n")
                f.write("\n")

                for reason, items in reason_groups.items():
                    f.write(f"Files deleted ({reason}):\n")
                    for item in items:
                        f.write(f"  - {item['file']}\n")
                    f.write("\n")
            else:
                f.write("No files were deleted.\n")

        print(f"📄 Report saved to: {output_file}")


# Usage
if __name__ == "__main__":
    # Initialize cleaner in DRY RUN mode first (safe)
    dry_run = True
    cleaner = TrustRAGLogsCleaner("logs", dry_run=dry_run)

    print("🔍 STEP 1: DRY RUN - Checking what would be deleted")
    deleted_files = cleaner.clean_all_folders()

    if dry_run:
        print("\n" + "⚠️ " * 20)
        print("⚠️  Ready to delete files? Change dry_run=True to dry_run=False")
        print("⚠️ " * 20)
    else:
        print("\n" + "⚠️ " * 20)
        print("⚠️  You are in LIVE DELETION mode!")
        print("⚠️ " * 20)

        # Uncomment the lines below to actually delete files
        # print("\n🗑️  STEP 2: ACTUAL DELETION")
        # cleaner_live = TrustRAGLogsCleaner("logs", dry_run=False)
        # cleaner_live.clean_all_folders()

    # Generate report
    cleaner.generate_report()
