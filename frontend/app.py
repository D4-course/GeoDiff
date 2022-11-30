from matplotlib.cbook import print_cycles
import pandas as pd
import streamlit as st
from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt
from matplotlib import style
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO, BytesIO
from PIL import Image
import base64

# f = io.BytesIO(base64.b64decode(b64string))
# pilimage = Image.open(f)

session = requests.Session()
retry = Retry(connect=10, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

style.use('ggplot')

st.set_page_config(page_title ="Conformation Generation App",initial_sidebar_state="expanded", layout="wide")
  
# change the color of button widget
Color = st.get_option("theme.secondaryBackgroundColor")
s = f"""
<style>
div.stButton > button:first-child {{background-color: #fffff ; border: 2px solid {Color}; border-radius:5px 5px 5px 5px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

def file_upload():
	FILE_TYPES = ["pkl"]
	file = st.file_uploader("Upload file", type=FILE_TYPES)
	show_file = st.empty()
	if not file:
		show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

	return file

def load_mol(file):
	lines = file.readlines()
	pmol = PandasMol2().read_mol2_from_list(mol2_lines=lines, mol2_code="2oc2_RX3_1_protein".encode())

	return pmol

def load_mol_from_lines(line):
	print(line)
	lines = line.split('\n')
	pmol = PandasMol2().read_mol2_from_list(mol2_lines=lines, mol2_code="2oc2_RX3_1_protein".encode())

	return pmol

def show_plot(pmol):
	fig = plt.figure()
	pmol.df['atom_type'].value_counts().plot(kind='bar')
	plt.xlabel('atom type')
	plt.ylabel('count')
	plt.title('Atom Types present in the molecule')
	st.pyplot(fig)
	
	plt.clf()

	pmol.df['element_type'] = pmol.df['atom_type'].apply(lambda x: x.split('.')[0])
	pmol.df['element_type'].value_counts().plot(kind='bar')
	plt.xlabel('element type')
	plt.ylabel('count')
	plt.title('Element Types present in the molecule')
	st.pyplot(fig)

	plt.clf()

	groupby_charge = pmol.df.groupby(['atom_type'])['charge']
	groupby_charge.mean().plot(kind='bar', yerr=groupby_charge.std())
	plt.ylabel('charge')
	plt.title('Average charge of different atom types')
	st.pyplot(fig)


def main():
	st.title("Generation of Conformations ")
	st.header("Conformation Generation using Dual Encoders ")
	st.write("This application allows us to generate conformations for molecules through the use of the diffusion process and iteratively denoising it.")
	
	activities = ["About this application", "Prediction"]
	st.sidebar.title("Navigation")
	choices = st.sidebar.radio("",activities)
	
	if choices == 'About this application':
		st.header("Context")
		st.write("Conformations are more intuitive and natural representations of molecules as compared to graph based methods. Conformations are key to\
				  Molecular Dynamics. However, most recent methods bypass atomic coordinates to use complicated methods and algorithms. The authors took inspiration\
				  from nonequilibrium thermodynamics to propose GeoDiff - a diffusion model - that far outperforms contemporary methods.")
		st.header("About this application")
		st.write("This webapp implements the program **GeoDiff** to generate conformations with high accuracy and diversity that can far outperform current standards.")
	elif choices == "Prediction":
		start_id = int(st.number_input('Insert the Start ID', key=0))
		end_id = int(st.number_input('Insert the End ID', key=1))
		file = file_upload()
		res_contents = []
		res_contents1 = []
		url = "http://localhost:80/uploadfiles"
		if file:
			files = {'files': file}
			print(files)
			payload = {'start': start_id, 'end': end_id}
			res = session.post(url, data=payload, files=files)
			print(res.json())
			if 'files' not in res.json() and 'files1' not in res.json():
				pass
			else:
				f = BytesIO(base64.b64decode((res.json()['files']['body'])))
				pilimage = Image.open(f)
				res_contents1 = res.json()['files1']
		if len(res_contents1):
			st.image(pilimage)
			df = pd.read_csv(StringIO(res_contents1))
			st.session_state.df = df
			# st.download_button('Download Conformations', res_contents, file_name='samples_all.pkl')
			st.download_button('Download Metrics', res_contents1, file_name='samples_all_covmat.csv')
			# Show Dataframe 
			st.text("Dataframe")
			st.dataframe(df)
			# show_plot(pmol)


if __name__ == '__main__':
	main()