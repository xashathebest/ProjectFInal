from app import db, app

with app.app_context():
    # Drop all tables first (if they exist)
    db.drop_all()
    
    # Create all tables
    db.create_all()
    
    print("Database initialized successfully!") 