from pymongo import *
import ssl

MONGODB_URI = "mongodb+srv://syedbasitabbas10:FZg3aL0FbRYyxGdh@topmedicalarticles.pfo2g.mongodb.net/?retryWrites=true&w=majority&appName=TopMedicalArticles"

try:
    client = MongoClient(
        MONGODB_URI,
        server_api=ServerApi('1'),
        tls=True,
        tlsAllowInvalidCertificates=True,
        ssl_version=ssl.PROTOCOL_TLSv1_2,
        socketTimeoutMS=30000,
        connectTimeoutMS=30000
    )
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
