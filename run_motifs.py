import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
# import sys
# sys.path.append(os.getcwd())

from signed_motif_detection import *


def main(session, directory_input, directory_output):
        # stimuli = ['natural_scene', 'natural_movie', 'drifting_gratings', 'static_gratings', 'spontaneous']
        stimuli = ['natural_scene','natural_movie','drifting_gratings']
        # stimuli = ['drifting_gratings']
        # model_choices = ['erdos_renyi', 'degree_preserving', 'pair_preserving', 'signed_pair_preserving']
        model_choices = ['signed_pair_preserving']

        for model in model_choices:
            intensity_df_list = []
            significant_ccg_list = []
            for stimulus in stimuli:

                data = np.load(f'{directory_input}/{session}_{stimulus}.npz')
                significant_ccg = data['significant_ccg']  # weights matrix with NaNs replaced by 0s
                # significant_ccg = significant_ccg[:100, :100]
                significant_confidence = data['significant_confidence']  # Load confidence matrix separately if available
                # significant_confidence = significant_confidence[:100, :100]
                significant_ccg_list.append(significant_ccg)
                # Create directed graph
                G = nx.from_numpy_array(significant_ccg, create_using=nx.DiGraph)

                input_G = G
                num_rewire = 10
                weight = 'confidence' # Z score of jitter-corrected CCG, can also be 'weight' for connection strength (jitter-corrected CCG) in the example graph. Use your own edge weight for your graph.
                cc = False
                Q = 100
                parallel = True # Set to True if you want to use multiprocessing to generate random graphs in parallel.
                num_cores = 23

                for i, j in zip(*np.nonzero(significant_ccg)):
                    G[i][j]['confidence'] = significant_confidence[i, j]

                random_graphs = random_graph_generator(input_G=input_G, num_rewire=num_rewire, model=model, weight=weight, cc=cc, Q=Q, parallel=parallel, num_cores=num_cores, disable=False)


                motif_types = ['021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300']
                motif_edges, motif_sms = {}, {}
                for motif_type in motif_types:
                    motif_edges[motif_type], motif_sms[motif_type] = get_edges_sms(motif_type, weight=weight)

                intensity_df = motif_census(G, random_graphs, all_signed_motif_types, motif_types, motif_edges, motif_sms, weight=weight, parallel=parallel, num_cores=num_cores)
                intensity_df.to_csv(f"{directory_output}/{session}_{stimulus}_intensity_df.csv", index=True)
                intensity_df_list.append(intensity_df)

            for i,stimulus in enumerate(stimuli):
                plot_motif_intensity_z_scores(intensity_df_list[i], f'{directory_output}/{model}_{stimulus}.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run motif detection with session and directory as input params.")
    parser.add_argument("--session", required=True, help="Session number")
    parser.add_argument("--directory_input", required=True, help="Directory containing input files")
    parser.add_argument("--directory_output", required=True, help="Directory containing output files")
    args = parser.parse_args()

    main(args.session, args.directory_input, args.directory_output)
