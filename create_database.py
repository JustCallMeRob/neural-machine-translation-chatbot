import sqlite3
import json
from datetime import datetime
import re
import time

timeframe = '2015-05'
sql_transaction = []

connection = sqlite3.connect('databases/{}.db'.format(timeframe))
cursor = connection.cursor()

# Create a table if one does not exist
def create_table():
    cursor.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, "
                   "parent TEXT, comment TEXT, subreddit TEXT, unix TEXT, score INT)")


# Convert the text of the comment body into a proper format
    # TODO: remove swear words
def preprocess_text(text):
    # Decontract any contracted words
    text = decontracted(text)
    # Replace all double quotes with single quotes
    text = text.replace('"', "'")
    # Remove links
    text = re.sub(r"http\S+", "", text)
    # Create space between punctuation and previous word. (eg: "Hello world." => "Hello world .")
    text = re.sub(r"([?.!,¿'])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    # Replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    # Remove trailing and leading whitespaces
    text = text.rstrip().strip()
    # Replace newline and return characters with our own symbols
    text = text.replace('\n', ' <nl> ').replace('\r', ' <nl> ')
    # Add tokens to indicate start and end of a comment
    text = '<start> ' + text + ' <end>'
    return text

# Replace word contractions in the comment body eg. "i'm alive" => "i am alive"
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Execute transaction builder
def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        cursor.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                cursor.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []

# Replace existing comments child comment with new comment that has higher score
def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, 
        unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))

# Insert comment with parent
def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES
         ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, time, score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))

# Insert comment with no parent
def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}",
        "{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, time, score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


# Limit the sentences to 10 words and 1000 characters long and ignore deleted or empty comments
def filter(data):
    if len(data.split(' ')) > 10 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '<start> deleted <end>':
        return False
    elif data == '<start> removed <end>':
        return False
    elif data == '<start>   <end>' or data == '<start>  <end>':
        return False
    else:
        return True

# Find parent reply in database based off of their id
def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        cursor.execute(sql)
        result = cursor.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        return False

# Find score in database based off of their parent reply id
def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        cursor.execute(sql)
        result = cursor.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        #print(str(e))
        return False


# Execute to create database
if __name__ == '__main__':

    create_table()

    comment_counter = 0
    parired_comment_counter = 0

    with open('D:/Datasets/reddit_data/{}/RC_{}'.format(timeframe.split('-')[0], timeframe), buffering=1000) as f:
        for row in f:
            comment_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id'].split('_')[1]
            body = preprocess_text(row['body'])
            score = row['score']
            try:
                comment_id = row['id'].split('_')[1]
            except Exception as e:
                comment_id = row['id']
            subreddit = row['subreddit']
            parent_data = find_parent(parent_id)
            utc = row['created_utc']

            if score >= 5:
                comment_score = find_existing_score(parent_id)
                if comment_score:
                    if score > comment_score:
                        if filter(body):
                            sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, utc, score)
                else:
                    if filter(body):
                        if parent_data:
                            sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, utc, score)
                            parired_comment_counter += 1
                        else:
                            sql_insert_no_parent(comment_id, parent_id, body, subreddit, utc, score)

            if comment_counter % 100000 == 0:
                print("Comments read: {}, Comments paired: {}".format(comment_counter, parired_comment_counter))
