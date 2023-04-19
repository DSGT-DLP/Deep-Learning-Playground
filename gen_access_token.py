import firebase_admin
from firebase_admin import auth

# initialize the Firebase app
cred = firebase_admin.credentials.Certificate('/path/to/service_account_key.json')
firebase_admin.initialize_app(cred)

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

# example usage
email = 'example@gmail.com'
token = generate_token(email)
print(token)
