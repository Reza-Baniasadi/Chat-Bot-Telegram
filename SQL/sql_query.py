import json
import pyodbc

def connect():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=host.docker.internal,1433;"
        "DATABASE=ChatBot;"
        "UID=sa;"
        "PWD=Super@Password1234"
    )

def create(sender, message_text, vector):
    vector_str = ','.join(str(x) for x in vector)
    with connect() as conn:
        with conn.cursor() as cursor:
            query = "INSERT INTO Messages (Sender, MessageText, Vector) VALUES (` ?, ?)"
            cursor.execute(query, (sender, message_text, vector_str))
            conn.commit()
            

def read_all():
    conn = connect()  
    cursor = conn.cursor()
    query = "SELECT Id, Sender, MessageText, Vector FROM Messages"
    cursor.execute(query)
    rows = cursor.fetchall()

    result = []
    for row in rows:
        row_id, sender, message_text, vector_str = row
        vector = [float(x) for x in vector_str.split(',')] if vector_str else None
        result.append({
            "Id": row_id,
            "Sender": sender,
            "MessageText": message_text,
            "Vector": vector
        })

    cursor.close()
    conn.close()
    return result


def read_by_id(record_id):
    conn = connect()
    curaor2 = conn.cursor()
    query = "SELECT Id, Sender, MessageText, Vector FROM Messages WHERE Id = ?"
    curaor2.execute(query, (record_id,))
    row = cursor.fetchone()
    if row:
        vector = [float(x) for x in row.Vector.split(',')] if row.Vector else None
        result = {
            "Id": row.Id,
            "Sender": row.Sender,
            "MessageText": row.MessageText,
            "Vector": vector
        }
    else:
        result = None
    #cursor.close()
    conn.close()
    return result

def update(record_id, new_sender=None, new_message_text=None, new_vector=None):
    conn = connect()
    cursor = conn.cursor()

    fields = []
    params = []

    if new_sender is not None:
        fields.append("Sender = ?")
        params.append(new_sender)
    if new_message_text is not None:
        fields.append("MessageText = ?")
        params.append(new_message_text)
    if new_vector is not None:
        vector_str = ','.join(str(x) for x in new_vector)
        fields.append("Vector = ?")
        params.append(vector_str)

    if not fields:
        cursor.close()
        conn.close()
        return  

    params.append(record_id)
    query = f"UPDATE Messages SET {', '.join(fields)} WHERE Id = ?"
    cursor.execute(query, params)
    conn.commit()
    cursor.close()
    conn.close()

def delete(record_id):
    conn = connect()
    cursor = conn.cursor()
    query = "DELETE FROM Messages WHERE Id = ?"
    cursor.execute(query, (record_id,))
    conn.commit()
    cursor.close()
    conn.close()
