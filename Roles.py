import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="velonix_db",
        user="postgres",
        password="nono4352"
    )

def add_user(conn):
    name = input("Enter user name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return

    with conn.cursor() as cur:
        cur.execute("SELECT name FROM roles WHERE name = %s", (name,))
        if cur.fetchone():
            print(f"Error: User '{name}' already exists")
            return

        print("\nEnter role status (y/n):")
        finance = input("Finance role (y/n): ").lower() == 'y'
        hr = input("HR role (y/n): ").lower() == 'y'
        it = input("IT role (y/n): ").lower() == 'y'

        cur.execute(
            """
            INSERT INTO roles (name, finance, hr, it) 
            VALUES (%s, %s, %s, %s) 
            RETURNING id
            """,
            (name, finance, hr, it)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        print(f"User {name} added successfully with ID: {user_id}")

def delete_user(conn):
    name = input("Enter the name of the user to delete: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return

    with conn.cursor() as cur:
        cur.execute("DELETE FROM roles WHERE name = %s RETURNING id", (name,))
        deleted_user = cur.fetchone()
        
        if deleted_user:
            conn.commit()
            print(f"User {name} deleted successfully!")
        else:
            print(f"Error: User '{name}' not found")

def edit_user_role(conn):
    name = input("Enter the name of the user to edit: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return

    with conn.cursor() as cur:
        while True:
            cur.execute(
                "SELECT finance, hr, it FROM roles WHERE name = %s",
                (name,)
            )
            user = cur.fetchone()
            
            if user:
                finance, hr, it = user
                print(f"\nCurrent roles for {name}:")
                print(f"Finance: {'Yes' if finance else 'No'}")
                print(f"HR: {'Yes' if hr else 'No'}")
                print(f"IT: {'Yes' if it else 'No'}")
                
                print("\nWhat role do you want to edit?")
                print("1. Finance")
                print("2. HR")
                print("3. IT")
                print("4. Quit")
                role = input("Enter the number of the role: ").strip()
                
                if role == '4':
                    print("Exiting role editor...")
                    break
                    
                if not role.isdigit() or role not in ['1', '2', '3']:
                    print("Error: Invalid role number")
                    continue
                    
                if role == '1':
                    finance = input("Finance role (y/n): ").lower() == 'y'
                elif role == '2':
                    hr = input("HR role (y/n): ").lower() == 'y'
                elif role == '3':
                    it = input("IT role (y/n): ").lower() == 'y'
                
                cur.execute(
                    """
                    UPDATE roles 
                    SET finance = %s, hr = %s, it = %s 
                    WHERE name = %s
                    """,
                    (finance, hr, it, name)
                )
                conn.commit()
                print("Role updated successfully!")
            else:
                print(f"Error: User '{name}' not found")
                break

def main():
    print("\n=== User Role Management System ===")
    
    while True:
        print("\nOptions:")
        print("1. Add User")
        print("2. Delete User")
        print("3. Edit User Role")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            print("Goodbye!")
            break
            
        conn = get_db_connection()
        if choice == '1':
            add_user(conn)
        elif choice == '2':
            delete_user(conn)
        elif choice == '3':
            edit_user_role(conn)
        else:
            print("Invalid choice! Please try again.")
        conn.close()

if __name__ == "__main__":
    main()
