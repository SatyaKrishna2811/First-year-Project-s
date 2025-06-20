import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

file_path=r"D:\medicine_dataset.csv"                      # assigning path to variable 


class Preprocessing:
    def __init__(self, file_path):
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', None)
        file_path=r"D:\medicine_dataset.csv"
        self.df = pd.read_csv(file_path)  # Load the dataset
        self.dc = self.df.drop_duplicates()  # Remove duplicates
        
    def view(self):
        print(self.df)  # Display the dataset
        print("Dataset shape:", self.df.shape)  # Print the shape of the dataset

    def drop_columns(self):
        for sub in range(2, 5):
            try:
                self.df.drop('substitute' + str(sub), axis=1, inplace=True)
            except:
                continue
        for sid in range(10, 42):
            try:
                self.df.drop('sideEffect' + str(sid), axis=1, inplace=True)
            except:
                continue
        for use in range(2, 5):
            try:
                self.df.drop('use' + str(use), axis=1, inplace=True)
            except:
                continue
        print(self.df.shape)

    def duplicates(self):
        print(self.df.duplicated())

    def null_objects(self):
        print(self.df.isnull())
        removing_null = self.dc.dropna()
        print(removing_null)

    def filtering(self):
        filter_by_nausea = self.df[self.df['sideEffect0'] == 'Nausea']
        print(filter_by_nausea)

    def filling_emptycells(self):
        self.dc['Chemical Class'] = self.dc['Chemical Class'].fillna('Unknown')

    def top_5(self):
        side_effect_columns = [col for col in self.df.columns if col.startswith('sideEffect')]
        side_effects = self.df[side_effect_columns].melt(value_name='SideEffect')
        side_effects = side_effects.dropna(subset=['SideEffect'])
        side_effect_counts = side_effects['SideEffect'].value_counts()
        side_effect_counts= side_effect_counts('Unknown')
        top_5_side_effects = side_effect_counts.head(5)
        print("The top 5 most common sideffects are:", top_5_side_effects)

    def class_count(self):
        class_counts = self.df.groupby('Action Class').size()
        print(class_counts)

class Visualization:
    def __init__(self, df):
        self.df = df

    def bar_graph(self):
        class_counts = self.df['Therapeutic Class'].value_counts()
        plt.bar(class_counts.index, class_counts.values)
        plt.xlabel('Therapeutic Class')
        plt.ylabel('Count of Drugs')
        plt.title('Count of Drugs by Therapeutic Class')
        plt.xticks(rotation=90)
        #plt.xlim(0,40)
        plt.show()

    def histogram(self):
        plt.hist(self.df['substitute0'].dropna())
        print(self.df['substitute0'].value_counts())
        plt.xlabel('Substitute Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Substitute Counts')
        plt.xticks(rotation=90)
        plt.xlim(1925,1950)                                              
        plt.show()

    def pie_chart(self):
        habit_counts = self.df['Habit Forming'].value_counts()
        plt.pie(habit_counts, labels=habit_counts.index, autopct='%.2f%%')
        plt.title('Percentage of Habit Forming Drugs')
        plt.show()

    def bar_comparison(self):
        chemical_class_counts = self.df['Chemical Class'].value_counts()
        plt.bar(chemical_class_counts.index, chemical_class_counts.values)
        plt.xlabel('Chemical Class')
        plt.ylabel('Frequency')
        plt.title('Frequency of Chemical Class Categories')
        plt.xticks(rotation=90)
        plt.show()

    def scatter_graph(self):
        substitute0_counts = self.df['substitute0'].value_counts()
        substitute1_counts = self.df['substitute1'].value_counts()
        stacked_data = pd.DataFrame({'Substitute0': substitute0_counts, 'Substitute1': substitute1_counts}).fillna(0)
        stacked_data = stacked_data.sort_index()
        stacked_data.plot(kind='area', stacked=False, alpha=0.2, figsize=(10, 6))
        plt.title('Cumulative Counts of Substitute0 and Substitute1')
        plt.xlabel('Categories')
        plt.ylabel('Counts')
        plt.legend(title='Substitute Types')
        plt.xticks(rotation=90)
        #plt.xlim(0,400)
        plt.show()

    def heatmap_before(self):
        sns.heatmap(self.df.isnull(), cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

    def heatmap_after(self):
        df_filled = self.df.fillna('Unknown')
        sns.heatmap(df_filled.isnull(), cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

    def filtered_plot(self):
        filter_by_nausea = self.df[self.df['sideEffect0'] == 'Nausea']
        tablet_counts = filter_by_nausea['name'].value_counts()
        top_15_tablets = tablet_counts.head(15)
        plt.figure(figsize=(8, 8))
        plt.pie(top_15_tablets, labels=top_15_tablets.index, autopct='%0.2f%%', colors=plt.cm.tab20.colors)
        plt.title('Top 15 Tablets Causing Nausea (Pie Chart)')
        plt.ylabel('')
        plt.show()
        plt.figure(figsize=(10, 6))
        top_15_tablets.plot(kind='bar', color='skyblue')
        plt.title('Top 15 Tablets Causing Nausea (Bar Graph)')
        plt.xlabel('Tablet Name')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def filtered_uses(self):
        filter_by_pain_relief = self.df[self.df['use0'] == ' Pain relief']
        tablet_counts = filter_by_pain_relief['name'].value_counts()
        top_15_tablets = tablet_counts.head(15)
        plt.figure(figsize=(8, 8))
        plt.pie(top_15_tablets, labels=top_15_tablets.index, autopct='%0.2f%%', colors=plt.cm.tab20.colors)
        plt.title('Top 15 Tablets curing Pain (Pie Chart)')
        plt.ylabel('')
        plt.show()
        plt.figure(figsize=(10, 6))
        top_15_tablets.plot(kind='bar', color='skyblue')
        plt.title('Top 15 Tablets curing Pain (Bar Graph)')
        plt.xlabel('Tablet Name')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def filtered_substitute(self):
        filter_by_substitute = self.df[self.df['substitute0'] == 'Solvin LS Syrup']
        tablet_counts = filter_by_substitute['name'].value_counts()
        top_15_tablets = tablet_counts.head(15)
        plt.figure(figsize=(8, 8))
        plt.pie(top_15_tablets, labels=top_15_tablets.index, autopct='%0.2f%%', colors=plt.cm.tab20.colors)
        plt.title('Top 15 Tablets with substitute as Solvin LS Syrup (Pie Chart)')
        plt.ylabel('')
        plt.show()
        plt.figure(figsize=(10, 6))
        top_15_tablets.plot(kind='bar', color='skyblue')
        plt.title('Top 15 Tablets with substitute as Solvin LS Syrup (Bar Graph)')
        plt.xlabel('Tablet Name')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def word_cloud_therapeutic(self):
        data1 = self.df['Therapeutic Class'].value_counts().to_dict()
        wc = WordCloud(width=800, height=400, background_color='white', max_font_size=300).generate_from_frequencies(data1)
        plt.figure(figsize=(20, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def word_cloud_sideeffect(self):
        data2 = self.df['sideEffect0'].value_counts().to_dict()
        wc = WordCloud(width=800, height=400, background_color="white", max_font_size=300).generate_from_frequencies(data2)
        plt.figure(figsize=(20, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def word_cloud_name(self):
        data2 = self.df['name'].value_counts().to_dict()
        wc = WordCloud(width=800, height=400, background_color="white", max_font_size=300).generate_from_frequencies(data2)
        plt.figure(figsize=(20, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def product_detail(self):
        side_name = input("Name of the tablet you want to search: ")
        def name(x, y):
            filter_by_name = x[x['name'] == y]
            print(filter_by_name)
            nameinput=pd.DataFrame(filter_by_name)
        name(self.df, side_name)

    def product_search(self):

        while True:
            pname = input('Enter the name of the tablet you want to search(type "exit" to quit): ')
            if pname == "exit":
                print("You have exited product search. Goodbye!")
                break
            elif pname in self.df['name'].values:
                print("The product you searched for is available: ", pname)
            else:
                print("The product you searched for, ", pname, ", is not available")


preprocessing=Preprocessing(file_path)
visualization=Visualization(preprocessing.df)

# preprocessing.view()
# preprocessing.drop_columns()
# preprocessing.duplicates()
# preprocessing.null_objects()
# preprocessing.filtering()
# preprocessing.filling_emptycells()
# preprocessing.top_5()
# preprocessing.class_count()

# visualization.bar_graph()
# visualization.histogram()
# visualization.pie_chart()
# visualization.bar_comparison()
# visualization.scatter_graph()
# visualization.heatmap_before()
# visualization.heatmap_after()
# visualization.filtered_plot()
# visualization.filtered_uses()
# visualization.filtered_substitute()
# visualization.word_cloud_therapeutic()
# visualization.word_cloud_sideeffect()
visualization.word_cloud_name()
# visualization.product_detail()
# visualization.product_search()
