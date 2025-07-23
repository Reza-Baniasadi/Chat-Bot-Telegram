def create(text, table_name):
    conn = connect()
    cursor = conn.cursor()
    query = f"INSERT INTO {table_name} (text) VALUES (?)"
    cursor.execute(query, (text,))
    conn.commit()
    cursor.close()
    conn.close()

def read_all(table_name):
    conn = connect()
    cursor = conn.cursor()
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def update(record_id, new_text, table_name):
    conn = connect()
    cursor = conn.cursor()
    query = f"UPDATE {table_name} SET text = ? WHERE id = ?"
    cursor.execute(query, (new_text, record_id))
    conn.commit()
    cursor.close()
    conn.close()


