import csv
import ast


def validate_csv_line(row):
    # Assuming the CSV format is:
    # id, annotation_id, bbox, keypoints, target
    try:
        # Validate bbox - should be a list of floats
        bbox = ast.literal_eval(row[2])
        if not isinstance(bbox, list) or not all(isinstance(x, (float, int)) for x in bbox):
            return False, "BBox format invalid"

        # Validate keypoints - should be a list of floats
        keypoints = ast.literal_eval(row[3])
        if not isinstance(keypoints, list) or not all(isinstance(x, (float, int)) for x in keypoints):
            return False, "Keypoints format invalid"

        # Validate target - should be a list of floats
        target = ast.literal_eval(row[4])
        if not isinstance(target, list) or not all(isinstance(x, (float, int)) for x in target):
            return False, "Target format invalid"

    except (ValueError, SyntaxError):
        return False, "Error parsing list from string"

    return True, "Valid"


def validate_csv_file(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            is_valid, message = validate_csv_line(row)
            if not is_valid:
                print(f"Line {i + 1} is invalid: {message}")
            else:
                print(f"Line {i + 1} is valid.")


# Example usage
validate_csv_file('../data/validation_data.csv')