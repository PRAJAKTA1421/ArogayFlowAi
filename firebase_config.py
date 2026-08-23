import os
import firebase_admin
from firebase_admin import credentials, firestore


def initialize_firebase():
    """
    Initialize Firebase Admin SDK and return Firestore client.
    """

    if firebase_admin._apps:
        return firestore.client()

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
        )

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Firebase credentials file not found: {credentials_path}"
        )

    cred = credentials.Certificate(credentials_path)

    firebase_admin.initialize_app(cred)

    return firestore.client()


db = initialize_firebase()