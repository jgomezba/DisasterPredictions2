import sys
import pandas as pd
import sqlite3

def count_words_from_string(text:str) -> int:
    """Simple function to count words splitted by space in a sentence

    Args:
        text (str): text to process

    Returns:
        int: Number of words
    """
    return len(text.split(" "))

def get_short_long_message(text:str) -> str:
    """Function to obtain if a message is short or long

    Args:
        text (str): text to measure

    Returns:
        str: decision if a message is short or long
    """
    words_number = count_words_from_string(text)
    
    if words_number > 30:
        return "Long message"
    else:
        return "Short message"

def load_data(messages_filepath:str, categories_filepath:str) -> pd.DataFrame:
    """Read csvs to analysis given paths

    Args:
        messages_filepath (str): path for messages csv
        categories_filepath (str): path for categories for messages csv

    Returns:
        pd.DataFrama: return merged dataframe between both csvs
    """
    #Read csvs
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    #Merge csvs
    df_merged = df_messages.merge(df_categories, on = ["id"], how="inner")
    
    return df_merged


def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """Function to clean data to have the desired format to database

    Args:
        df (pd.DataFrame): dataframe to clean

    Returns:
        pd.DataFrame: dataframe clean
    """
    #Drop duplicates if exist
    df.drop_duplicates(inplace=True)
    
    # Split the categories column into separate category-value pairs
    categories_split = df['categories'].str.split(';', expand=True)
    
    # Extract category names from the first row (before the dash)
    row = categories_split.iloc[0]
    category_colnames = row.str.split('-').str[0]

    # Create a new DataFrame for categories
    categories_expanded = categories_split.apply(lambda col: col.str.split('-').str[1])

    # Convert the values to integers
    categories_expanded = categories_expanded.astype(int)

    # Rename the columns with the category names
    categories_expanded.columns = category_colnames

    # Drop the original categories column and concatenate the new columns
    df = pd.concat([df.drop('categories', axis=1), categories_expanded], axis=1)
    
    return df

def add_new_metrics(df:pd.DataFrame):
    df['Message length'] = df['message'].apply(count_words_from_string)
    df['ShortLong message'] = df['message'].apply(get_short_long_message)
    
    return df
    
def save_data(df:pd.DataFrame, database_filename:str):
    """Function to save dataframe clean into a sqlite database

    Args:
        df (pd.DataFrame): dataframe to save
        database_filename (str): filename to database where dataframe will be saved
    """
    #Connect to sqlite
    conn = sqlite3.connect(database_filename)
    
    #Try to create database
    try:
        df.to_sql("data", conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Error saving dataframe in database: {e}")
    finally:
        conn.close()    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Adding new columns for graphs')
        df = add_new_metrics(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()