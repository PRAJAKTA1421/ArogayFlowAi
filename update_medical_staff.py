import random
from datetime import datetime

from firebase_config import db


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def distribute_staff(total_staff):
    """
    Split total staff into:
        Doctors
        Nurses
        Paramedics

    The distribution is synthetic demo data.
    """

    total_staff = max(
        3,
        int(total_staff)
    )

    # Rough operational distribution
    doctors = max(
        1,
        round(total_staff * 0.22)
    )

    nurses = max(
        1,
        round(total_staff * 0.48)
    )

    paramedics = (
        total_staff
        - doctors
        - nurses
    )

    if paramedics < 1:

        paramedics = 1

        nurses = max(
            1,
            total_staff - doctors - paramedics
        )

    # Make sure everything adds up
    difference = (
        total_staff
        - doctors
        - nurses
        - paramedics
    )

    paramedics += difference

    return {
        "doctors": doctors,
        "nurses": nurses,
        "paramedics": paramedics,
    }


def generate_role_data(
    total_staff,
    role_total,
    role_name,
    rng
):
    """
    Generate structured staffing information
    for one staff category.

    This is synthetic demo operational data.
    """

    # Some PHCs have additional required capacity.
    shortage_probability = 0.25

    if rng.random() < shortage_probability:

        extra_required = rng.randint(
            1,
            max(1, round(role_total * 0.30))
        )

    else:

        extra_required = 0

    required = role_total + extra_required

    # Staff availability between roughly 65% and 100%
    minimum_present = max(
        1,
        int(role_total * 0.65)
    )

    present = rng.randint(
        minimum_present,
        role_total
    )

    remaining = role_total - present

    # Split non-present staff into leave/unavailable
    if remaining > 0:

        on_leave = rng.randint(
            0,
            remaining
        )

        unavailable = (
            remaining
            - on_leave
        )

    else:

        on_leave = 0
        unavailable = 0

    gap = max(
        0,
        required - present
    )

    return {
        "required": int(required),
        "total": int(role_total),
        "present": int(present),
        "on_leave": int(on_leave),
        "unavailable": int(unavailable),
        "gap": int(gap),
    }


# ============================================================
# UPDATE PHC STAFF
# ============================================================

def update_medical_staff():

    docs = list(
        db.collection("phcs").stream()
    )

    docs.sort(
        key=lambda doc: doc.id
    )

    print(
        f"Found {len(docs)} PHC documents."
    )

    if not docs:

        print(
            "No PHCs found. Nothing to update."
        )

        return


    batch = db.batch()

    batch_count = 0

    updated_count = 0


    for index, doc in enumerate(docs):

        data = doc.to_dict() or {}

        phc_id = doc.id

        # ----------------------------------------------------
        # Existing staff count
        # ----------------------------------------------------

        existing_total = safe_int(
            data.get(
                "total_staff",
                0
            )
        )

        existing_present = safe_int(
            data.get(
                "present_staff",
                0
            )
        )


        # Safety defaults
        if existing_total < 3:

            existing_total = 12


        existing_present = max(
            0,
            min(
                existing_present,
                existing_total
            )
        )


        # ----------------------------------------------------
        # Deterministic random generator
        # ----------------------------------------------------

        # Different but reproducible values per PHC
        seed_value = (
            RANDOM_SEED
            + sum(
                ord(char)
                for char in phc_id
            )
        )

        rng = random.Random(
            seed_value
        )


        # ----------------------------------------------------
        # Role distribution
        # ----------------------------------------------------

        role_totals = distribute_staff(
            existing_total
        )


        # ----------------------------------------------------
        # Generate role-level data
        # ----------------------------------------------------

        doctors = generate_role_data(
            existing_total,
            role_totals["doctors"],
            "doctors",
            rng
        )

        nurses = generate_role_data(
            existing_total,
            role_totals["nurses"],
            "nurses",
            rng
        )

        paramedics = generate_role_data(
            existing_total,
            role_totals["paramedics"],
            "paramedics",
            rng
        )


        # ----------------------------------------------------
        # Adjust role present values so that the overall
        # present_staff remains close to our existing database
        # value.
        # ----------------------------------------------------

        target_present = existing_present

        role_data = [
            doctors,
            nurses,
            paramedics,
        ]


        current_present = sum(
            role["present"]
            for role in role_data
        )


        # Difference between existing PHC availability
        # and generated role availability
        difference = (
            target_present
            - current_present
        )


        if difference > 0:

            for role in role_data:

                available_capacity = (
                    role["total"]
                    - role["present"]
                )

                increase = min(
                    difference,
                    available_capacity
                )

                role["present"] += increase

                # Recalculate leave/unavailable
                non_present = (
                    role["total"]
                    - role["present"]
                )

                role["on_leave"] = min(
                    role["on_leave"],
                    non_present
                )

                role["unavailable"] = (
                    non_present
                    - role["on_leave"]
                )

                role["gap"] = max(
                    0,
                    role["required"]
                    - role["present"]
                )

                difference -= increase

                if difference <= 0:
                    break


        elif difference < 0:

            reduction = abs(
                difference
            )

            for role in role_data:

                reducible = max(
                    0,
                    role["present"] - 1
                )

                decrease = min(
                    reduction,
                    reducible
                )

                role["present"] -= decrease

                non_present = (
                    role["total"]
                    - role["present"]
                )

                role["on_leave"] = min(
                    role["on_leave"],
                    non_present
                )

                role["unavailable"] = (
                    non_present
                    - role["on_leave"]
                )

                role["gap"] = max(
                    0,
                    role["required"]
                    - role["present"]
                )

                reduction -= decrease

                if reduction <= 0:
                    break


        # ----------------------------------------------------
        # Final totals
        # ----------------------------------------------------

        total_staff = sum(
            role["total"]
            for role in role_data
        )

        present_staff = sum(
            role["present"]
            for role in role_data
        )

        on_leave = sum(
            role["on_leave"]
            for role in role_data
        )

        unavailable = sum(
            role["unavailable"]
            for role in role_data
        )

        required_staff = sum(
            role["required"]
            for role in role_data
        )

        gap = max(
            0,
            required_staff - present_staff
        )

        availability_percent = (
            (
                present_staff
                / total_staff
            ) * 100
            if total_staff > 0
            else 0
        )


        # ----------------------------------------------------
        # Structured Firestore document
        # ----------------------------------------------------

        staff_data = {

            "doctors": doctors,

            "nurses": nurses,

            "paramedics": paramedics,

            "summary": {

                "required":
                    required_staff,

                "total":
                    total_staff,

                "present":
                    present_staff,

                "on_leave":
                    on_leave,

                "unavailable":
                    unavailable,

                "gap":
                    gap,

                "availability_percent":
                    round(
                        availability_percent,
                        1
                    ),
            },
        }


        update = {

            # Preserve existing fields
            "total_staff":
                total_staff,

            "present_staff":
                present_staff,

            # New structured staff data
            "staff":
                staff_data,

            # Explicitly identify demo data
            "staff_data_source":
                "synthetic_demo",

            "staff_schema_version":
                "1.0",

            "staff_updated_at":
                datetime.utcnow(),
        }


        batch.update(
            doc.reference,
            update
        )

        batch_count += 1
        updated_count += 1


        # Firestore batch limit safety
        if batch_count >= 400:

            batch.commit()

            print(
                f"✓ Committed batch "
                f"({updated_count} PHCs)"
            )

            batch = db.batch()

            batch_count = 0


        print(
            f"✓ {phc_id} | "
            f"staff {present_staff}/{total_staff} | "
            f"gap {gap} | "
            f"availability "
            f"{availability_percent:.1f}%"
        )


    # --------------------------------------------------------
    # Final batch
    # --------------------------------------------------------

    if batch_count > 0:

        batch.commit()

        print(
            f"✓ Final batch committed "
            f"({batch_count} PHCs)"
        )


    print()
    print(
        "=" * 60
    )

    print(
        "MEDICAL STAFF DATA UPDATE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"PHCs updated: {updated_count}"
    )

    print(
        "Schema version: 1.0"
    )

    print(
        "Data source: synthetic_demo"
    )

    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    update_medical_staff()