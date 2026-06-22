import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

def process_rotation_data(data_dir):
    data_dir = os.path.join(data_dir, "parallel_csvs")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    cumulative_df = pd.DataFrame()
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        data = pd.read_csv(file_path)
        cumulative_df = pd.concat([cumulative_df, data], ignore_index=True)
    cumulative_df['response'] = cumulative_df['response'].apply(lambda x: x.strip('[]').strip("'").strip('{}').lower())
    cumulative_df['ground_truth'] = cumulative_df['ground_truth'].str.lower()
    cumulative_df['correct'] = (cumulative_df['response'] == cumulative_df['ground_truth']).astype(int)
    return cumulative_df

class ResponseAnalysis:
    def __init__(self, df, plot_dir):
        self.df = df
        self.plot_dir = plot_dir
        self.conf_matrix = self.calculate_confusion_matrix()

    def analyze_responses(self):
        print(f"Accuracy: {self._calculate_measure(self.df, 'accuracy')}")
        print(f"Recall: {self._calculate_measure(self.df, 'recall')}")
        print(f"Precision: {self._calculate_measure(self.df, 'precision')}")
        print(f"F1 Score: {self._calculate_measure(self.df, 'f1_score')}")
        print(f"Confusion Matrix: {self.conf_matrix}")

    def _calculate_recall(self, df):
        tp = len(df[(df['ground_truth'] == 'yes') & (df['response'] == 'yes')])
        fn = len(df[(df['ground_truth'] == 'yes') & (df['response'] == 'no')])
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return recall

    def _calculate_precision(self, df):
        tp = len(df[(df['ground_truth'] == 'yes') & (df['response'] == 'yes')])
        fp = len(df[(df['ground_truth'] == 'no') & (df['response'] == 'yes')])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        return precision
    
    def _calculate_f1_score(self, df):
        precision = self._calculate_precision(df)
        recall = self._calculate_recall(df)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score

    def calculate_confusion_matrix(self):
        confusion_matrix = {
            'TP': len(self.df[(self.df['ground_truth'] == 'yes') & (self.df['response'] == 'yes')]),
            'TN': len(self.df[(self.df['ground_truth'] == 'no') & (self.df['response'] == 'no')]),
            'FP': len(self.df[(self.df['ground_truth'] == 'no') & (self.df['response'] == 'yes')]),
            'FN': len(self.df[(self.df['ground_truth'] == 'yes') & (self.df['response'] == 'no')])
        }
        return confusion_matrix
    
    def _calculate_measure(self, df, measure="recall"):
        if measure == "accuracy":
            measure_result = df['correct'].mean()
        elif measure == "recall":
            measure_result = self._calculate_recall(df)
        elif measure == "precision":
            measure_result = self._calculate_precision(df)
        elif measure == "f1_score":
            measure_result = self._calculate_f1_score(df)
        return measure_result
    
    def _plot_by_dict(self, measure_dict, measure, separator, width=0.8, group=False):
        plt.bar(measure_dict.keys(), measure_dict.values(), width=width)
        plt.xticks(list(measure_dict.keys()))
        plt.xlabel(separator)
        plt.ylabel(measure.capitalize())
        plt.title(f'PAC {measure.capitalize()} by {separator}')
        plt.savefig(os.path.join(plot_dir, f'{measure.lower()}_by_{separator.lower()}{"_group" if group else ""}.png'))
        plt.close()
        print(f"{measure.capitalize()} by {separator}]: {measure_dict}")

    def plot_measure_by_script(self, measure="recall"):
        measure_by_script = {script: self._calculate_measure(self.df[self.df['script'] == script], measure=measure) for script in self.df['script'].unique()}
        self._plot_by_dict(measure_by_script, measure, separator="Script")

class RotationResponseAnalysis(ResponseAnalysis):
    def __init__(self, df, plot_dir):
        super().__init__(df, plot_dir)

    def plot_measure_by_angle(self, measure="recall", group=False):
        if group:
            angle_groups = [(0, 0), (10, 30), (40, 60), (70, 90)]
            measure_by_angle = {}
            for angle_range in angle_groups:
                angle_df = self.df[(self.df['angle'] >= angle_range[0]) & (self.df['angle'] <= angle_range[1])]
                measure_result = self._calculate_measure(angle_df, measure=measure)
                measure_by_angle[f"{angle_range[0]}-{angle_range[1]}"] = measure_result
            width = 0.8
        else:
            measure_by_angle = {angle.item(): self._calculate_measure(self.df[self.df['angle'] == angle], measure=measure) for angle in self.df['angle'].unique()}
            width = 6.0
        self._plot_by_dict(measure_by_angle, measure, separator="Angle", width=width, group=group)

    def plot_measure_by_angle_and_script(self, measure="recall"):
        angles = sorted(self.df['angle'].unique())
        scripts = self.df['script'].unique()
        accuracy_matrix = {script: [] for script in scripts}

        for angle in angles:
            for script in scripts:
                subset = self.df[(self.df['angle'] == angle) & (self.df['script'] == script)]
                measure_result = self._calculate_measure(subset, measure=measure)
                accuracy_matrix[script].append(measure_result)
        
        for script, accuracies in accuracy_matrix.items():
            plt.plot(angles, accuracies, marker='o', label=script)
        plt.xlabel('Rotation Angle')
        plt.ylabel(measure.capitalize())
        plt.title(f'PAC {measure.capitalize()} by Angle and Script')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{measure.lower()}_by_angle_and_script.png'))
        plt.close()
        print(f"Plotted {measure.capitalize()} by Angle and Script")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze rotation recognition using a vision-language model.")
    parser.add_argument("--plot_dir", type=str, help="The directory to save plots.")
    parser.add_argument("--data_dir", type=str, help="The data directory.")
    parser.add_argument("--model", type=str, help="The model name.")
    parser.add_argument("--prompt_name", type=str, help="The prompt name.")
    
    args = parser.parse_args()
    #prompt_name = "prompt_identity"
    # plot_dir should be set via constructor/argument
    #os.makedirs(plot_dir, exist_ok=True)
    # data_dir should be set via constructor/argument
    df = process_rotation_data(args.data_dir)
    df.to_csv(os.path.join(args.data_dir, f"qwen2.5_vl_{args.prompt_name}_all.csv"), index=False)
    print("Saved combined CSV file.")
    analysis = ResponseAnalysis(df, args.plot_dir)
    analysis.analyze_responses()
    