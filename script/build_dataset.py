import csv
import json
from pathlib import Path
from collections import Counter


# ============================================================
# 1. File paths
# ============================================================

# build_dataset.py is inside HW1/script/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "original"

TEST_FILE = ORIGINAL_DIR / "test.csv"
INTENT_LABELS_FILE = ORIGINAL_DIR / "intent_labels.txt"
SLOT_LABELS_FILE = ORIGINAL_DIR / "slot_labels.txt"

OUTPUT_FILE = DATA_DIR / "evaluation_dataset.json"


# ============================================================
# 2. Selected examples
#
# Each intent has exactly 4 examples:
# 7 intents x 4 examples = 28 total examples.
#
# The indices are 0-based row indices in test.csv,
# excluding the CSV header.
# ============================================================

SELECTED_BY_INTENT = {
    "AddToPlaylist": [
        0,
        21,
        62,
        68,
    ],

    "BookRestaurant": [
        6,
        17,
        48,
        51,
    ],

    "GetWeather": [
        3,
        38,
        39,
        45,
    ],

    "PlayMusic": [
        4,
        32,
        44,
        79,
    ],

    "RateBook": [
        20,
        56,
        72,
        105,
    ],

    "SearchCreativeWork": [
        14,
        16,
        35,
        80,
    ],

    "SearchScreeningEvent": [
        10,
        58,
        223,
        368,
    ],
}


# ============================================================
# 3. SNIPS slot types that count as an explicit location
# ============================================================

LOCATION_SLOT_TYPES = {
    "city",
    "state",
    "country",
    "current_location",
    "geographic_poi",
    "location_name",
    "poi",
}


def load_text_labels(path):
    """
    Load labels from a text file.
    Each non-empty line is treated as one label.
    """
    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def load_test_dataset():
    """
    Load all rows from test.csv.

    Expected columns:
        input
        intent
        slots
    """
    with open(TEST_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required_columns = {"input", "intent", "slots"}

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            f"test.csv must contain columns: {required_columns}. "
            f"Found: {reader.fieldnames}"
        )

    return rows


def get_slot_type(slot_label):
    """
    Convert BIO slot labels into their base slot type.

    Examples:
        B-city  -> city
        I-city  -> city
        O       -> None
    """
    if slot_label == "O":
        return None

    if slot_label.startswith("B-") or slot_label.startswith("I-"):
        return slot_label[2:]

    return slot_label


def mentions_location(slots):
    """
    Derive the boolean ground truth for:

        "Does the utterance explicitly mention a location?"

    Returns True if at least one slot is a location-related slot.
    """
    slot_labels = slots.split()

    for label in slot_labels:
        slot_type = get_slot_type(label)

        if slot_type in LOCATION_SLOT_TYPES:
            return True

    return False


def validate_labels(intent_labels, slot_labels):
    """
    Make sure the intents and location slot types we use actually
    exist in the original SNIPS label files.
    """

    # Validate intents
    for intent in SELECTED_BY_INTENT:
        if intent not in intent_labels:
            raise ValueError(
                f"Intent '{intent}' was not found in intent_labels.txt"
            )

    # Remove B-/I- prefixes from slot labels
    available_slot_types = {
        get_slot_type(label)
        for label in slot_labels
        if label != "O"
    }

    # Validate location slot types
    missing_location_types = (
        LOCATION_SLOT_TYPES - available_slot_types
    )

    if missing_location_types:
        raise ValueError(
            "Some location slot types were not found in "
            f"slot_labels.txt: {missing_location_types}"
        )


def build_evaluation_dataset(rows):
    """
    Extract the 28 selected examples and create the evaluation dataset.
    """
    examples = []

    used_indices = set()

    for expected_intent, indices in SELECTED_BY_INTENT.items():

        if len(indices) != 4:
            raise ValueError(
                f"{expected_intent} must contain exactly 4 examples."
            )

        for test_index in indices:

            # Prevent duplicated examples
            if test_index in used_indices:
                raise ValueError(
                    f"Duplicate test index found: {test_index}"
                )

            used_indices.add(test_index)

            # Make sure index exists
            if test_index >= len(rows):
                raise IndexError(
                    f"test_index {test_index} is outside the dataset."
                )

            row = rows[test_index]

            utterance = row["input"].strip()
            actual_intent = row["intent"].strip()
            slots_string = row["slots"].strip()
            slots_list = slots_string.split()

            # Make sure the selected row still has the expected intent.
            # This protects us if the dataset changes in the future.
            if actual_intent != expected_intent:
                raise ValueError(
                    f"Intent mismatch at test index {test_index}: "
                    f"expected '{expected_intent}', "
                    f"but found '{actual_intent}'."
                )

            location_answer = mentions_location(slots_string)

            example = {
                "id": f"snips_test_{test_index:03d}",

                "state": {
                    "utterance": utterance
                },

                "source": {
                    "dataset": "bkonkle/snips-joint-intent",
                    "split": "test",
                    "test_index": test_index
                },

                "source_labels": {
                    "intent": actual_intent,
                    "slots": slots_list
                },

                "expected": {
                    "intent": actual_intent,
                    "mentions_location": location_answer
                }
            }

            examples.append(example)

    return examples


def validate_output(examples):
    """
    Perform simple checks on the final evaluation dataset.
    """

    if len(examples) != 28:
        raise ValueError(
            f"Expected 28 examples, found {len(examples)}."
        )

    # Count intents
    intent_counts = Counter(
        example["expected"]["intent"]
        for example in examples
    )

    # Every intent should have exactly four examples
    for intent, count in intent_counts.items():
        if count != 4:
            raise ValueError(
                f"{intent} has {count} examples instead of 4."
            )

    # Count boolean labels
    location_counts = Counter(
        example["expected"]["mentions_location"]
        for example in examples
    )

    return intent_counts, location_counts


def main():

    print("Loading original SNIPS dataset...")

    rows = load_test_dataset()

    intent_labels = load_text_labels(INTENT_LABELS_FILE)
    slot_labels = load_text_labels(SLOT_LABELS_FILE)

    print(f"Loaded {len(rows)} test examples.")

    # Check original labels
    validate_labels(intent_labels, slot_labels)

    # Create the 28-example evaluation dataset
    examples = build_evaluation_dataset(rows)

    # Validate result
    intent_counts, location_counts = validate_output(examples)

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            examples,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"Created: {OUTPUT_FILE}")
    print(f"Total examples: {len(examples)}")

    print("\nIntent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")

    print("\nLocation boolean distribution:")
    print(f"  True:  {location_counts.get(True, 0)}")
    print(f"  False: {location_counts.get(False, 0)}")


if __name__ == "__main__":
    main()