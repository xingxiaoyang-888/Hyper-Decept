import sqlite3
from neo4j import GraphDatabase

# 1. Read follow relationships from SQLite
def read_follow_data():
    conn = sqlite3.connect('./data/simu_db/yaml_200/time0.db')
    cursor = conn.cursor()
    cursor.execute("SELECT follower_id, followee_id FROM follow")
    follow_data = cursor.fetchall()
    conn.close()
    return follow_data

# 2. Neo4j database connection configuration
uri = "bolt://localhost:7687"
user = "neo4j"
password = "neo4j"
driver = GraphDatabase.driver(uri, auth=(user, password))

# 3. Function to create nodes and relationships
def create_graph(tx, follow_data):
    # Collect all node IDs
    all_ids = set()
    for row in follow_data:
        all_ids.add(row[0])
        all_ids.add(row[1])

    # Split IDs into Good and Bad agents
    good_ids = [id for id in range(0, 900)]
    bad_ids = [id for id in range(900, 1000)]

    # Batch create Good nodes
    if good_ids:
        tx.run("""
        UNWIND $ids AS id
        MERGE (:Good:Agent {id: id})
        """, ids=good_ids)
    
    # Batch create Bad nodes
    if bad_ids:
        tx.run("""
        UNWIND $ids AS id
        MERGE (:Bad:Agent {id: id})
        """, ids=bad_ids)

    # Batch create follow relationships
    tx.run("""
    UNWIND $rows AS row
    MATCH (follower:Agent {id: row.follower_id})
    MATCH (followee:Agent {id: row.followee_id})
    MERGE (follower)-[:FOLLOWS]->(followee)
    """, rows=[{"follower_id": row[0], "followee_id": row[1]} for row in follow_data])

# 4. Execute data import
def import_data():
    follow_data = read_follow_data()
    with driver.session() as session:
        session.execute_write(create_graph, follow_data)
    print("Data import complete!")

# 5. Query subgraph of Good agents following Bad agents
def query_good_to_bad():
    with driver.session() as session:
        result = session.run("""
        MATCH (follower:Good)-[r:FOLLOWS]->(followee:Bad)
        RETURN follower.id AS follower_id, 
               followee.id AS followee_id,
               count(r) AS relationship_count
        """)
        
        print("\nStatistics of Good agents following Bad agents:")
        for record in result:
            print(f"Good {record['follower_id']} → Bad {record['followee_id']}")

# Main execution
if __name__ == "__main__":
    import_data()
    query_good_to_bad()
    driver.close()