# Homework Spark Fundamentals
# Date: 2024-12-13
# Name: Tarik Bel Attar
# Import necessary PySpark modules and Python standard libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col
import logging
import sys

# ========================================
# Configure Logging
# ========================================

# Create a logger object
logger = logging.getLogger("PySparkLogger")
logger.setLevel(logging.INFO)  # Set the logging level to INFO

# Create a console handler to output logs to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Set handler level to INFO

# Define a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# ========================================
# Define Helper Functions
# ========================================

def load_dataframe(spark, file_path, alias_name, df_description):
    """
    Loads a CSV file into a Spark DataFrame with error handling and logging.

    Parameters:
    - spark (SparkSession): The SparkSession object.
    - file_path (str): The path to the CSV file.
    - alias_name (str): The alias to assign to the DataFrame.
    - df_description (str): A descriptive name for the DataFrame (used in logs).

    Returns:
    - DataFrame or None: Returns the loaded DataFrame if successful, else None.
    """
    try:
        logger.info(f"Starting to load DataFrame: {df_description} from {file_path}")

        # Read the CSV file into a DataFrame
        df = (
            spark.read
            .option("header", "true")          # Indicates that the first line in the CSV file contains headers
            .option("inferSchema", "true")     # Instructs Spark to automatically infer data types of columns
            .csv(file_path)                     # Path to the CSV file
            .alias(alias_name)                 # Assign an alias for easier reference in joins
        )

        # Log success and schema
        logger.info(f"Successfully loaded DataFrame: {df_description}")
        df.printSchema()

        return df

    except Exception as e:
        logger.error(f"Error loading DataFrame: {df_description} from {file_path}")
        logger.error(f"Exception: {e}")
        return None

def validate_dataframe(df, df_description):
    """
    Validates that the DataFrame is not None and contains data.

    Parameters:
    - df (DataFrame or None): The DataFrame to validate.
    - df_description (str): A descriptive name for the DataFrame (used in logs).

    Returns:
    - bool: True if DataFrame is valid, False otherwise.
    """
    if df is None:
        logger.error(f"DataFrame {df_description} is None. Skipping further processing.")
        return False
    elif df.rdd.isEmpty():
        logger.warning(f"DataFrame {df_description} is empty.")
        return False
    else:
        logger.info(f"DataFrame {df_description} is loaded and contains data.")
        return True

# ========================================
# Initialize SparkSession
# ========================================

def initialize_spark(app_name="homework_spark_fundamental"):
    """
    Initializes and returns a SparkSession.

    Parameters:
    - app_name (str): The name of the Spark application.

    Returns:
    - SparkSession: The initialized SparkSession object.
    """
    try:
        logger.info("Initializing SparkSession.")
        spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        logger.info("SparkSession initialized successfully.")
        return spark
    except Exception as e:
        logger.error("Failed to initialize SparkSession.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as SparkSession is essential

# ========================================
# Main Execution Flow
# ========================================

def main():
    """
    Main function to execute the data loading, table creation, and data insertion processes.
    """
    # Initialize SparkSession
    spark = initialize_spark()

    # Define file paths and metadata
    data_files = {
        "match_details": {
            "path": "/home/iceberg/data/match_details.csv",
            "alias": "md",
            "description": "Match Details"
        },
        "matches": {
            "path": "/home/iceberg/data/matches.csv",
            "alias": "m",
            "description": "Matches"
        },
        "medals_matches_players": {
            "path": "/home/iceberg/data/medals_matches_players.csv",
            "alias": "mmp",
            "description": "Medals Matches Players"
        },
        "medals": {
            "path": "/home/iceberg/data/medals.csv",
            "alias": "med",
            "description": "Medals"
        },
        "maps": {
            "path": "/home/iceberg/data/maps.csv",
            "alias": "maps",
            "description": "Maps"
        }
    }

    # Load all DataFrames
    dataframes = {}
    for df_key, df_info in data_files.items():
        df = load_dataframe(
            spark,
            file_path=df_info["path"],
            alias_name=df_info["alias"],
            df_description=df_info["description"]
        )
        dataframes[df_key] = df

    # Validate DataFrames
    for df_key, df in dataframes.items():
        validate_dataframe(df, data_files[df_key]["description"])

    # Proceed only if all required DataFrames are loaded successfully
    required_dfs = ["match_details", "matches", "medals_matches_players", "medals", "maps"]
    all_valid = all([validate_dataframe(dataframes[df], data_files[df]["description"]) for df in required_dfs])

    if not all_valid:
        logger.error("One or more required DataFrames are missing or empty. Exiting the script.")
        sys.exit(1)  # Exit the script as essential DataFrames are missing

    # ========================================
    # Create Iceberg Tables
    # ========================================

    # Define DDL statements for each Iceberg table
    ddl_statements = {
        "match_details": """
            CREATE TABLE IF NOT EXISTS bootcamp.match_details(
                match_id STRING,
                player_gamertag STRING,
                previous_spartan_rank INTEGER,
                spartan_rank INTEGER,
                previous_total_xp INTEGER,
                total_xp INTEGER,
                previous_csr_tier INTEGER,
                previous_csr_designation INTEGER,
                previous_csr INTEGER,
                previous_csr_percent_to_next_tier INTEGER,
                previous_csr_rank INTEGER,
                current_csr_tier INTEGER,
                current_csr_designation INTEGER,
                current_csr INTEGER,
                current_csr_percent_to_next_tier INTEGER,
                current_csr_rank INTEGER,
                player_rank_on_team INTEGER,
                player_finished BOOLEAN,
                player_average_life STRING,
                player_total_kills INTEGER,
                player_total_headshots INTEGER,
                player_total_weapon_damage DOUBLE,
                player_total_shots_landed INTEGER,
                player_total_melee_kills INTEGER,
                player_total_melee_damage DOUBLE,
                player_total_assassinations INTEGER,
                player_total_ground_pound_kills INTEGER,
                player_total_shoulder_bash_kills INTEGER,
                player_total_grenade_damage DOUBLE,
                player_total_power_weapon_damage DOUBLE,
                player_total_power_weapon_grabs INTEGER,
                player_total_deaths INTEGER,
                player_total_assists INTEGER,
                player_total_grenade_kills INTEGER,
                did_win INTEGER,
                team_id INTEGER
            ) USING iceberg
            PARTITIONED BY (bucket(16, match_id))
        """,
        "matches": """
            CREATE TABLE IF NOT EXISTS bootcamp.matches(
                match_id STRING, 
                mapid STRING, 
                is_team_game BOOLEAN, 
                playlist_id STRING, 
                game_variant_id STRING, 
                is_match_over BOOLEAN, 
                completion_date TIMESTAMP, 
                match_duration STRING, 
                game_mode STRING, 
                map_variant_id STRING 
            ) USING iceberg
            PARTITIONED BY (bucket(16, match_id))
        """,
        "medal_matches_players": """
            CREATE TABLE IF NOT EXISTS bootcamp.medal_matches_players(
                match_id STRING,
                player_gamertag STRING,
                medal_id LONG,
                count INTEGER
            ) USING iceberg
            PARTITIONED BY (bucket(16, match_id))
        """,
        "maps": """
            CREATE TABLE IF NOT EXISTS bootcamp.maps(
                mapid STRING,
                name STRING,
                description STRING
            ) USING iceberg
            PARTITIONED BY (mapid)
        """,
        "medals": """
            CREATE TABLE IF NOT EXISTS bootcamp.medals(
                medal_id LONG,
                sprite_uri STRING,
                sprite_left INTEGER,
                sprite_top INTEGER,
                sprite_sheet_width INTEGER,
                sprite_sheet_height INTEGER,
                sprite_width INTEGER,
                sprite_height INTEGER,
                classification STRING,
                description STRING,
                name STRING,
                difficulty INTEGER
            ) USING iceberg
            PARTITIONED BY (medal_id)
        """
    }

    # Iterate over each DDL statement and execute it
    for table_name, ddl in ddl_statements.items():
        try:
            logger.info(f"Creating Iceberg table: bootcamp.{table_name}")
            spark.sql(ddl)
            logger.info(f"Iceberg table bootcamp.{table_name} created successfully.")
        except Exception as e:
            logger.error(f"Error creating Iceberg table bootcamp.{table_name}")
            logger.error(f"Exception: {e}")
            sys.exit(1)  # Exit the script as table creation failed

    # ========================================
    # Insert Data into Iceberg Tables
    # ========================================

    # Insert data into bootcamp.match_details
    try:
        logger.info("Inserting data into Iceberg table: bootcamp.match_details")
        dataframes["match_details"].write.mode("overwrite").saveAsTable("bootcamp.match_details")
        logger.info("Data inserted successfully into bootcamp.match_details")
    except Exception as e:
        logger.error("Error inserting data into bootcamp.match_details")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as data insertion failed

    # Insert data into bootcamp.matches
    try:
        logger.info("Inserting data into Iceberg table: bootcamp.matches")
        dataframes["matches"].write.mode("overwrite").saveAsTable("bootcamp.matches")
        logger.info("Data inserted successfully into bootcamp.matches")
    except Exception as e:
        logger.error("Error inserting data into bootcamp.matches")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as data insertion failed

    # Insert data into bootcamp.medal_matches_players
    try:
        logger.info("Inserting data into Iceberg table: bootcamp.medal_matches_players")
        dataframes["medals_matches_players"].write.mode("overwrite").saveAsTable("bootcamp.medal_matches_players")
        logger.info("Data inserted successfully into bootcamp.medal_matches_players")
    except Exception as e:
        logger.error("Error inserting data into bootcamp.medal_matches_players")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as data insertion failed

    # Insert data into bootcamp.maps
    try:
        logger.info("Inserting data into Iceberg table: bootcamp.maps")
        dataframes["maps"].write.mode("overwrite").saveAsTable("bootcamp.maps")
        logger.info("Data inserted successfully into bootcamp.maps")
    except Exception as e:
        logger.error("Error inserting data into bootcamp.maps")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as data insertion failed

    # Insert data into bootcamp.medals
    try:
        logger.info("Inserting data into Iceberg table: bootcamp.medals")
        dataframes["medals"].write.mode("overwrite").saveAsTable("bootcamp.medals")
        logger.info("Data inserted successfully into bootcamp.medals")
    except Exception as e:
        logger.error("Error inserting data into bootcamp.medals")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as data insertion failed

    # ========================================
    # Terminate SparkSession
    # ========================================
    try:
        logger.info("Terminating SparkSession.")
        spark.stop()
        logger.info("SparkSession terminated successfully.")
    except Exception as e:
        logger.error("Error occurred while terminating SparkSession.")
        logger.error(f"Exception: {e}")

# ========================================
# Execute Main Function
# ========================================

if __name__ == "__main__":
    main()