import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression
from scripts.energy_cost_model import train_energy_cost_model, predict_energy_cost
from scripts.shortest_path import create_weighted_graph, find_shortest_path
import logging
import argparse

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Input values
    parser = argparse.ArgumentParser(description="Calculate shortest path and energy cost.")
    parser.add_argument("--body_mass", type=float, default=75, help="Body mass of the person (in kg)")
    parser.add_argument("--points_resolution", type=float, default=10, help="Resolution of altitude points (in meters)")
    parser.add_argument("--walking_speed", type=float, default=50, help="Walking speed (in meters/min)")
    args = parser.parse_args()

     # Input values
    body_mass_input = args.body_mass
    points_resolution_input = args.points_resolution
    walking_speed_input = args.walking_speed
    
    lodge_entrance_coords = (200, 559)

    # Log input values
    logging.info(f"Body Mass: {body_mass_input} kg")
    logging.info(f"Points Resolution: {points_resolution_input} meters")
    logging.info(f"Walking Speed: {walking_speed_input} meters/min")



    # READING FILES
    logging.info("Reading input csv files: energy_cost.csv and altitude_map.csv")
    df_energy = pd.read_csv('data/energy_cost.csv')
    df_altitude = pd.read_csv('data/altitude_map.csv')
    logging.info("Files read successfully.")

    # ETL
    # Create a list starting from 0, increasing by 10, and containing 683 elements
    x_list_index = [i for i in range(len(df_altitude.columns.values))]
    y_list_index = [i for i in range(len(df_altitude))]
    df_altitude.columns = x_list_index

    # MODELLING
    logging.info("Training energy cost model...")
    model_positive, model_negative = train_energy_cost_model(df_energy)
    logging.info("Energy cost model trained successfully.")


    # OPTIMIZING
    logging.info("Creating weighted graph...")
    G = create_weighted_graph(model_positive, model_negative, df_altitude, body_mass_input, points_resolution_input, walking_speed_input)
    logging.info("Weighted graph created successfully.")


    # Find shortest path and total energy
    logging.info("Finding shortest path...")
    shortest_path, total_energy = find_shortest_path(model_positive, model_negative, G, lodge_entrance_coords, df_altitude, body_mass_input, points_resolution_input, walking_speed_input)

    if shortest_path is not None:
        # Get latitude and longitude values from DataFrame
        x_vals = df_altitude.columns.astype(float)
        y_vals = df_altitude.index.values.astype(float)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid of latitude and longitude values
        X, Y = np.meshgrid(x_vals, y_vals)
        # Get altitude values to plot
        Z = df_altitude.values
        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

        # Set labels for axes
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Altitude (meters)')

        # Set title
        ax.set_title(f"Altitude Map: beginner's trail (approx energy cost = {round(total_energy,2)} Joules)")

        ax.set_zlim(0, 1.5 * Z.max())

        # Set the view angle
        elevation_angle = 45
        azimuth_angle = -110
        ax.view_init(elev=elevation_angle, azim=azimuth_angle)

        # Extract X, Y, and Z coordinates from the shortest path
        shortest_path_x = [coord[0] for coord in shortest_path]
        shortest_path_y = [coord[1] for coord in shortest_path]
        shortest_path_z = [df_altitude.at[y, x] for x, y in shortest_path]


        # Create a DataFrame with X and Y coordinates
        shortest_path_df = pd.DataFrame({'x_coord': shortest_path_x, 'y_coord': shortest_path_y})

        # Multiply each value in the DataFrame by 10
        shortest_path_df = shortest_path_df.apply(lambda x: x * points_resolution_input)
        # Invert the row order using iloc[::-1]
        shortest_path_df = shortest_path_df.iloc[::-1]
        # Save the DataFrame to a CSV file with column headings
        csv_filename = f'shortest_path_coords_mass{body_mass_input}_presol{points_resolution_input}_wspeed{walking_speed_input}.csv'
        shortest_path_df.to_csv(f'results/{csv_filename}', index=False)

        # Plot the shortest path as a line on the 3D plot
        ax.plot(shortest_path_x, shortest_path_y, shortest_path_z, color='red', label='beginner’s trail', linewidth=3, linestyle='dashed')

        # Add a red point at the lodge entrance
        x_red_point = lodge_entrance_coords[0]
        y_red_point = lodge_entrance_coords[1]
        z_red_point = df_altitude.at[y_red_point, x_red_point]  # Get the altitude at the red point
        ax.scatter(x_red_point, y_red_point, z_red_point, color='red', s=100, label='Lodge-Entrance')

        # Multiplicar por 10 los valores de las marcas en los ejes x e y para obtener los nuevos nombres de las marcas
        ticks_x = plt.xticks()[0]
        ticks_y = plt.yticks()[0]
        new_tick_labels_x = [str(int(tick * points_resolution_input)) for tick in ticks_x]
        new_tick_labels_y = [str(int(tick * points_resolution_input)) for tick in ticks_y]

        # Establecer los nuevos nombres de las marcas en los ejes x e y
        plt.xticks(ticks_x, new_tick_labels_x)
        plt.yticks(ticks_y, new_tick_labels_y)

        # Establecer los límites de los ejes x e y
        plt.xlim(ticks_x[1], ticks_x[-2])
        plt.ylim(ticks_y[1], ticks_y[-2])

        # Añadir una leyenda al gráfico
        ax.legend()

        # Save the plot as a PNG file
        plt.savefig(f'results/map_beginners_trail_mass{body_mass_input}_presol{points_resolution_input}_wspeed{walking_speed_input}.png', bbox_inches='tight', dpi=300)

        # Show the plot
        #plt.show()
    else:
        print("No reachable points found along the specified coordinate line.")
