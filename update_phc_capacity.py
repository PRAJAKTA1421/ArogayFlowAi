import random
from datetime import datetime

from firebase_config import db

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def main():
    docs = list(db.collection("phcs").stream())
    docs.sort(key=lambda d: d.id)

    print(f"Found {len(docs)} PHC documents.")

    for doc in docs:
        data = doc.to_dict() or {}

        # Keep existing PHC identity/location/demand data unchanged.
        # Only fill the missing capacity/staff fields needed by the dashboard.
        total_beds = data.get("total_beds")
        occupied_beds = data.get("occupied_beds")
        available_beds = data.get("available_beds")
        total_staff = data.get("total_staff")
        present_staff = data.get("present_staff")

        if total_beds is None:
            total_beds = random.randint(20, 120)

        if occupied_beds is None:
            occupied_beds = random.randint(
                int(total_beds * 0.35),
                int(total_beds * 0.92),
            )

        if available_beds is None:
            available_beds = max(total_beds - occupied_beds, 0)

        if total_staff is None:
            total_staff = random.randint(12, 40)

        if present_staff is None:
            present_staff = random.randint(
                max(5, int(total_staff * 0.65)),
                total_staff,
            )

        update = {
            "total_beds": int(total_beds),
            "occupied_beds": int(occupied_beds),
            "available_beds": int(available_beds),
            "total_staff": int(total_staff),
            "present_staff": int(present_staff),
            "updated_at": datetime.utcnow(),
        }

        db.collection("phcs").document(doc.id).update(update)
        print(
            f"✓ {doc.id}: beds {occupied_beds}/{total_beds}, "
            f"available {available_beds}, staff {present_staff}/{total_staff}"
        )

    print("\nPHC capacity/staff data update complete.")


if __name__ == "__main__":
    main()
