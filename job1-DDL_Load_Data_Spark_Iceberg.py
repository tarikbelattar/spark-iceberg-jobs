# Homework Spark Fundamentals
# Date: 2024-12-13
# Name: Tarik Bel Attar
# Import necessary PySpark modules and Python standard libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, avg, desc, broadcast, countDistinct
import logging
import sys

# ========================================
# Configure Logging
# ========================================

# Create a logger object
logger = logging.getLogger("AggregationJobLogger")
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
# Initialize SparkSession
# ========================================

def initialize_spark(app_name="aggregation_spark_job"):
    """
    Initializes and returns a SparkSession with specific configurations.

    Parameters:
    - app_name (str): The name of the Spark application.

    Returns:
    - SparkSession: The initialized SparkSession object.
    """
    try:
        logger.info("Initializing SparkSession.")
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .getOrCreate()
        logger.info("SparkSession initialized successfully.")
        return spark
    except Exception as e:
        logger.error("Failed to initialize SparkSession.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as SparkSession is essential

# ========================================
# Load Iceberg Tables
# ========================================

def load_iceberg_tables(spark):
    """
    Loads Iceberg tables into Spark DataFrames.

    Parameters:
    - spark (SparkSession): The active SparkSession.

    Returns:
    - dict: A dictionary containing loaded DataFrames.
    """
    try:
        logger.info("Loading Iceberg tables into DataFrames.")

        # Load Iceberg tables with aliases
        match_details = spark.table("bootcamp.match_details").alias("md")
        matches = spark.table("bootcamp.matches").alias("m")
        medal_matches_players = spark.table("bootcamp.medal_matches_players").alias("mmp")
        medals = spark.table("bootcamp.medals").alias("med")
        maps = spark.table("bootcamp.maps").alias("map")

        logger.info("Successfully loaded all Iceberg tables.")

        return {
            "match_details": match_details,
            "matches": matches,
            "medal_matches_players": medal_matches_players,
            "medals": medals,
            "maps": maps
        }

    except Exception as e:
        logger.error("Error loading Iceberg tables.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as table loading failed

# ========================================
# Explicitly Broadcast 'medals' and 'maps'
# ========================================

def broadcast_tables(dataframes):
    """
    Explicitly broadcasts the 'medals' and 'maps' DataFrames.

    Parameters:
    - dataframes (dict): Dictionary containing the loaded DataFrames.

    Returns:
    - dict: Dictionary containing the broadcasted DataFrames.
    """
    try:
        logger.info("Broadcasting 'medals' and 'maps' tables.")

        # Explicitly broadcast 'medals' and 'maps'
        medals_broadcast = broadcast(dataframes["medals"])
        maps_broadcast = broadcast(dataframes["maps"])

        logger.info("'medals' and 'maps' tables have been broadcasted successfully.")

        return {
            "medals_broadcast": medals_broadcast,
            "maps_broadcast": maps_broadcast
        }

    except Exception as e:
        logger.error("Error broadcasting tables.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as broadcasting failed

# ========================================
# Perform Joins with Corrected Column Naming
# ========================================

def perform_joins(dataframes, broadcasts):
    """
    Performs the required joins with specified configurations.

    Parameters:
    - dataframes (dict): Dictionary containing loaded DataFrames.
    - broadcasts (dict): Dictionary containing broadcasted DataFrames.

    Returns:
    - DataFrame: The joined DataFrame with unique column names.
    """
    try:
        logger.info("Starting join operations.")

        # Step 1: Join 'match_details' (md) with 'matches' (m) on 'match_id'
        joined_df = dataframes["match_details"].join(
            dataframes["matches"],
            on="match_id",
            how="inner"
        ).select(
            "match_id",
            "player_gamertag",       # From md
            "player_total_kills",    # From md
            "playlist_id",           # From m
            "mapid"                  # From m
        )
        logger.info("Joined 'match_details' with 'matches' on 'match_id'.")

        # Step 2: Join the above with 'medal_matches_players' (mmp) on 'match_id'
        # Exclude 'player_gamertag' from mmp to avoid duplication
        joined_df = joined_df.join(
            dataframes["medal_matches_players"].select(
                "match_id",
                "medal_id",
                "count"
            ),
            on="match_id",
            how="inner"
        ).select(
            "match_id",
            "player_gamertag",
            "player_total_kills",
            "playlist_id",
            "mapid",
            "medal_id",
            "count"
        )
        logger.info("Joined with 'medal_matches_players' on 'match_id'.")

        # Step 3: Join with 'medals' (med) on 'medal_id' using the broadcasted DataFrame
        # Rename 'name' to 'medal_name' and 'count' to 'medal_count' to avoid ambiguity
        joined_df = joined_df.join(
            broadcasts["medals_broadcast"],
            on="medal_id",
            how="inner"
        ).select(
            "match_id",
            "player_gamertag",
            "player_total_kills",
            "playlist_id",
            "mapid",
            "medal_id",
            "count",
            "name"  # From med
        ).withColumnRenamed("name", "medal_name").withColumnRenamed("count", "medal_count")
        logger.info("Joined with 'medals' on 'medal_id' and renamed columns to resolve ambiguity.")

        # Step 4: Join with 'maps' (map) on 'mapid' using the broadcasted DataFrame
        # Rename 'name' to 'map_name' to avoid ambiguity
        joined_df = joined_df.join(
            broadcasts["maps_broadcast"],
            on="mapid",
            how="inner"
        ).select(
            "match_id",
            "player_gamertag",
            "player_total_kills",
            "playlist_id",
            "mapid",
            "medal_id",
            "medal_count",
            "medal_name",
            "name"  # From map
        ).withColumnRenamed("name", "map_name")
        logger.info("Joined with 'maps' on 'mapid' and renamed columns to resolve ambiguity.")

        logger.info("All join operations completed successfully with unique column names.")

        return joined_df

    except Exception as e:
        logger.error("Error during join operations.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as joins failed

# ========================================
# Aggregate Data
# ========================================

def aggregate_data(joined_df):
    """
    Aggregates the joined DataFrame to answer specific analytical questions.

    Parameters:
    - joined_df (DataFrame): The joined DataFrame.

    Returns:
    - dict: A dictionary containing aggregated results.
    """
    try:
        logger.info("Starting data aggregation.")

        # 1. Which player averages the most kills per game?
        kills_per_game = joined_df.groupBy("player_gamertag") \
            .agg(avg("player_total_kills").alias("avg_kills_per_game")) \
            .orderBy(desc("avg_kills_per_game"))

        top_player_kills = kills_per_game.first()
        if top_player_kills:
            logger.info(f"Top Player (Most Kills per Game): {top_player_kills['player_gamertag']} with {top_player_kills['avg_kills_per_game']:.2f} kills/game.")
        else:
            logger.info("No data available for player kills per game.")

        # 2. Which playlist gets played the most?
        playlist_popularity = joined_df.groupBy("playlist_id") \
            .agg(countDistinct("match_id").alias("play_count")) \
            .orderBy(desc("play_count"))

        top_playlist = playlist_popularity.first()
        if top_playlist:
            logger.info(f"Top Playlist (Most Played): {top_playlist['playlist_id']} with {top_playlist['play_count']} plays.")
        else:
            logger.info("No data available for playlist popularity.")

        # 3. Which map gets played the most?
        map_popularity = joined_df.groupBy("mapid") \
            .agg(countDistinct("match_id").alias("map_count")) \
            .orderBy(desc("map_count"))

        top_map = map_popularity.first()
        if top_map:
            logger.info(f"Top Map (Most Played): {top_map['mapid']} with {top_map['map_count']} plays.")
        else:
            logger.info("No data available for map popularity.")

        # 4. Which map do players get the most Killing Spree medals on?
        # Assuming "Killing Spree" medals have the name "Killing Spree"
        killing_spree_medals = joined_df.filter(col("medal_name") == "Killing Spree")
        killing_spree_medals_distinct = killing_spree_medals.dropDuplicates(["match_id"])

        killing_spree_map = killing_spree_medals_distinct.groupBy("mapid") \
            .agg(expr("SUM(medal_count) as total_killing_spree_medals")) \
            .orderBy(desc("total_killing_spree_medals"))

        top_killing_spree_map = killing_spree_map.first()
        if top_killing_spree_map:
            logger.info(f"Top Killing Spree Map: {top_killing_spree_map['mapid']} with {top_killing_spree_map['total_killing_spree_medals']} medals.")
        else:
            logger.info("No data available for Killing Spree medals.")

        # Collect all results
        results = {
            "top_player_kills": top_player_kills,
            "top_playlist": top_playlist,
            "top_map": top_map,
            "top_killing_spree_map": top_killing_spree_map
        }

        return results

    except Exception as e:
        logger.error("Error during data aggregation.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as aggregation failed

# ========================================
# Experiment with sortWithinPartitions
# ========================================

def experiment_sort_within_partitions(joined_df):
    """
    Experiments with different sortWithinPartitions to analyze data size impacts.

    Parameters:
    - joined_df (DataFrame): The joined DataFrame.

    Returns:
    - None
    """
    try:
        logger.info("Starting experiments with sortWithinPartitions.")

        # Function to calculate total data size in bytes after sorting
        def calculate_data_size(df_sorted):
            return df_sorted.rdd.map(lambda row: len(str(row))).sum()

        # Example 1: Sort by 'playlist_id'
        sorted_by_playlist = joined_df.sortWithinPartitions("playlist_id")
        size_by_playlist = calculate_data_size(sorted_by_playlist)
        logger.info(f"Total data size after sorting within partitions by 'playlist_id': {size_by_playlist} bytes.")

        # Example 2: Sort by 'mapid'
        sorted_by_map = joined_df.sortWithinPartitions("mapid")
        size_by_map = calculate_data_size(sorted_by_map)
        logger.info(f"Total data size after sorting within partitions by 'mapid': {size_by_map} bytes.")

        # Example 3: Sort by 'player_gamertag'
        sorted_by_player = joined_df.sortWithinPartitions("player_gamertag")
        size_by_player = calculate_data_size(sorted_by_player)
        logger.info(f"Total data size after sorting within partitions by 'player_gamertag': {size_by_player} bytes.")

        # Example 4: Sort by 'match_id'
        sorted_by_match = joined_df.sortWithinPartitions("match_id")
        size_by_match = calculate_data_size(sorted_by_match)
        logger.info(f"Total data size after sorting within partitions by 'match_id': {size_by_match} bytes.")

        # Summary of sizes
        logger.info("Summary of sortWithinPartitions experiments:")
        logger.info(f"Sort by 'playlist_id': {size_by_playlist} bytes")
        logger.info(f"Sort by 'mapid': {size_by_map} bytes")
        logger.info(f"Sort by 'player_gamertag': {size_by_player} bytes")
        logger.info(f"Sort by 'match_id': {size_by_match} bytes")

    except Exception as e:
        logger.error("Error during sortWithinPartitions experiments.")
        logger.error(f"Exception: {e}")
        sys.exit(1)  # Exit the script as experiments failed

# ========================================
# Main Execution Flow
# ========================================

def main():
    """
    Main function to execute the aggregation Spark job.
    """
    # Initialize SparkSession
    spark = initialize_spark()

    # Load Iceberg tables
    dataframes = load_iceberg_tables(spark)

    # Broadcast 'medals' and 'maps' tables
    broadcasts = broadcast_tables(dataframes)

    # Perform joins with corrected column naming
    joined_df = perform_joins(dataframes, broadcasts)

    # Aggregate data to answer analytical questions
    results = aggregate_data(joined_df)

    # Display aggregated results
    logger.info("Aggregated Results:")
    if results["top_player_kills"]:
        logger.info(f"Top Player (Most Kills per Game): {results['top_player_kills']['player_gamertag']} with {results['top_player_kills']['avg_kills_per_game']:.2f} kills/game.")
    else:
        logger.info("No data available for Top Player.")

    if results["top_playlist"]:
        logger.info(f"Top Playlist (Most Played): {results['top_playlist']['playlist_id']} with {results['top_playlist']['play_count']} plays.")
    else:
        logger.info("No data available for Top Playlist.")

    if results["top_map"]:
        logger.info(f"Top Map (Most Played): {results['top_map']['mapid']} with {results['top_map']['map_count']} plays.")
    else:
        logger.info("No data available for Top Map.")

    if results["top_killing_spree_map"]:
        logger.info(f"Top Killing Spree Map: {results['top_killing_spree_map']['mapid']} with {results['top_killing_spree_map']['total_killing_spree_medals']} medals.")
    else:
        logger.info("No data available for Top Killing Spree Map.")

    # Experiment with sortWithinPartitions to analyze data size impacts
    experiment_sort_within_partitions(joined_df)

    # Terminate SparkSession
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