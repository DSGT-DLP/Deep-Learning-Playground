import firebase_admin
import argparse
from firebase_admin import auth

# initialize the Firebase app
config = {
  'apiKey': "AIzaSyAMJgYSG_TW7CT_krdWaFUBLxU4yRINxX8",
  'authDomain': "deep-learning-playground-8d2ce.firebaseapp.com",
  'projectId': "deep-learning-playground-8d2ce",
  'storageBucket': "deep-learning-playground-8d2ce.appspot.com",
  'messagingSenderId': "771338023154",
  'appId': "1:771338023154:web:8ab6e73fc9c646426a606b",
};


cred = firebase_admin.credentials.ApplicationDefault()
firebase_admin.initialize_app(credential=cred, options=config)

# function to generate a UID for a user based on their email
def generate_uid(email):
    user = auth.get_user_by_email(email)
    if user:
        return user.uid
    else:
        new_user = auth.create_user(email=email)
        return new_user.uid

# function to generate a Firebase authentication token (bearer token)
def generate_token(email):
    uid = generate_uid(email)
    token = auth.create_custom_token(uid)
    return token

def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate bearer token for backend API testing")
    parser.add_argument('--email', type=str, help='Email Address to generate bearer token for')
    args = parser.parse_args()
    email = args.email
    return email 

if __name__ == '__main__':
    email = parse_args()
    token = generate_token(email)
    print(f"your token is {token}")
