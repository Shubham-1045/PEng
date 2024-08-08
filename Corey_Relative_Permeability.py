import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
from PyPDF2 import PdfReader
import io
from io import BytesIO


# Function to calculate relative permeabilities and capillary pressure
def calculate_properties_water_oil(Sw, Swc, Sorw, Kro_max, krw_max, no, nw, pc_max, npc):
    kro = Kro_max * np.maximum(((1 - Sw - Sorw) / (1 - Swc - Sorw)), 0)**no
    krw = krw_max * np.maximum(((Sw - Swc) / (1 - Swc - Sorw)), 0)**nw
    pcwo = pc_max * np.maximum(((1 - Sw - Sorw) / (1 - Swc - Sorw)), 0)**npc
    return kro, krw, pcwo

def calculate_properties_gas_oil(Sg, Sgc, Sl, Kro_max, krg_max, ng, ngo, pc_max, npg):
    kro = Kro_max * np.maximum(((1 - Sg - Sl) / (1 - Sgc - Sl)), 0)**ngo
    krg = krg_max * np.maximum(((Sg - Sgc) / (1 - Sl - Sgc)), 0)**ng
    pcgo = pc_max * np.maximum(((Sg - Sgc) / (1 - Sl - Sgc)), 0)**npg
    return kro, krg, pcgo

def calculate_properties_gas_water(Sg, Sw, Sgc, Swc, Krg_max, krw_max, ng, nw, pc_max, npc):
    krg = Krg_max * np.maximum(((Sg - Sgc) / (1 - Swc - Sgc)), 0)**ng
    krw = krw_max * np.maximum(((Sw - Swc) / (1 - Swc)), 0)**nw
    pcgw = pc_max * np.maximum(((Sg - Sgc) / (1 - Swc - Sgc)), 0)**npc
    return krg, krw, pcgw

def export_to_ecl(system, saturation, kro, krw_or_krg, pc):
    lines = []

    if system == 'Water-Oil System':
        
        lines.append("SWOF")
        lines.append("--" + "Sw" + "\t" + "Krw" + "\t" + "Krow" + "\t" + "Pcow")
        for sw, krw, krow, pcow in zip(saturation, krw_or_krg, kro, pc):
            lines.append(f" {sw:.4f} \t{krw:.4f} \t{krow:.4f} \t{pcow:.4f}")
        lines.append("/")
    
    elif system == 'Gas-Oil System':
        
        lines.append("SGOF")
        lines.append("--" + "Sg" + "\t" + "Krg" + "\t" + "Krog" + "\t" + "Pcog")
        for sg, krg, krog, pcog in zip(saturation, krw_or_krg, kro, pc):
            s_complement = 1.0 - sg
            lines.append(f" {s_complement:.4f} \t{krg:.4f} \t{krog:.4f} \t{pcog:.4f}")
        lines.append("/")
    
    elif system == 'Gas-Water System':
        
        lines.append("SGWFN")
        lines.append("--" + "Sg" + "\t" + "Krg" + "\t" + "Krwg" + "\t" + "Pcwg")
        for sg, krg, krwg, pcwg in zip(saturation, kro, krw_or_krg, pc):
            lines.append(f" {sg:.4f} \t{krg:.4f} \t{krwg:.4f} \t{pcwg:.4f}")
        lines.append("/")

    ecl_data = "\n".join(lines)
    return ecl_data

# Function to plot relative permeabilities and capillary pressure
def plot_properties(system, x_values, y_values_1, y_values_2):
    fig = go.Figure()
    if system == 'Water-Oil System':
        x_label = 'Water Saturation (Sw)'
        y1_label = 'Oil Relative Permeability (kro)'
        y2_label = 'Water Relative Permeability (krw)'
        title = 'Water-Oil System Relative Permeability'
        color1 = 'green'
        color2 = 'blue'
    elif system == 'Gas-Oil System':
        x_label = 'Liquid Saturation (Sl)'
        y1_label = 'Gas Relative Permeability (krg)'
        y2_label = 'Oil Relative Permeability (kro)'
        title = 'Gas-Oil Relative Permeability'
        color1 = 'red'
        color2 = 'green'
    elif system == 'Gas-Water System':
        x_label = 'Water Saturation (Sw)'
        y1_label = 'Gas Relative Permeability (krg)'
        y2_label = 'Water Relative Permeability (krw)'
        title = 'Gas-Water Relative Permeability'
        color1 = 'red'
        color2 = 'blue'

    # Clamp y_values to range [0, 1]
    y_values_1 = np.clip(y_values_1, 0, 1)
    y_values_2 = np.clip(y_values_2, 0, 1)

    fig.add_trace(go.Scatter(x=x_values, y=y_values_1, mode='lines', name=y1_label, line=dict(color=color1)))
    fig.add_trace(go.Scatter(x=x_values, y=y_values_2, mode='lines', name=y2_label, line=dict(color=color2)))
    fig.add_trace(go.Scatter(x=[x_values[0], x_values[-1]], y=[y_values_1[0], y_values_1[-1]], mode='markers', name=f'{y1_label} end points', marker=dict(color=color1, size=8)))
    fig.add_trace(go.Scatter(x=[x_values[0], x_values[-1]], y=[y_values_2[0], y_values_2[-1]], mode='markers', name=f'{y2_label} end points', marker=dict(color=color2, size=8)))

    fig.update_layout(
        title=title,
        title_x=0,  # Align title to the left
        title_y=1,  # Position title at the top
        xaxis_title=x_label,
        yaxis_title='Relative Permeability',
        xaxis=dict(range=[0, 1], dtick=0.2),
        yaxis=dict(range=[0, 1]),  # Fix y-axis between 0 and 1
        showlegend=False,  # Hide legend
        margin=dict(l=20, r=20, t=50, b=20),  # Reduce margins
        height=400,  # Plot height
        width=600,  # Plot width
        title_xanchor='left',  # Anchor title to the left
        title_yanchor='top'  # Anchor title to the top
    )

    return fig

def plot_capillary_pressure(system, S, pc):
    # Ensure the values are sorted by saturation
    sorted_indices = np.argsort(S)
    S = S[sorted_indices]
    pc = pc[sorted_indices]

    fig = go.Figure()
    if system == 'Water-Oil System':
        x_label = 'Water Saturation (Sw)'
        title = 'Water-Oil Capillary Pressure'
    elif system == 'Gas-Oil System':
        x_label = 'Liquid Saturation (Sl)'
        title = 'Gas-Oil Capillary Pressure'
    elif system == 'Gas-Water System':
        x_label = 'Water Saturation (Sw)'
        title = 'Gas-Water Capillary Pressure'

    if len(S) > 0:
        fig.add_trace(go.Scatter(x=S, y=pc, mode='lines', name='Capillary Pressure (pc)', line=dict(color='dimgray')))
        fig.add_trace(go.Scatter(x=[S[0], S[-1]], y=[pc[0], pc[-1]], mode='markers', name='pc end points', marker=dict(color='dimgray', size=8)))
    
    fig.update_layout(
        title=title,
        title_x=0,  # Align title to the left
        title_y=1,  # Position title at the top
        xaxis_title=x_label,
        yaxis_title='Capillary Pressure (psi)',
        xaxis=dict(range=[0, 1], dtick=0.2),
        yaxis=dict(range=[0, max(pc)] if len(pc) > 0 else [0, 1], dtick=5.0),
        showlegend=False,  # Hide legend
        margin=dict(l=20, r=20, t=50, b=20),  # Reduce margins
        height=400,  # Plot height
        width=600,  # Plot width
        title_xanchor='left',  # Anchor title to the left
        title_yanchor='top'  # Anchor title to the top
    )

    return fig

def generate_plots_and_data(system, params):
    if system == 'Water-Oil System':
        Sw = np.linspace(params['Swc'], 1 - params['Sorw'], 100)
        kro, krw, pcwo = calculate_properties_water_oil(Sw, params['Swc'], params['Sorw'], params['Kro_max'], params['krw_max'], params['no'], params['nw'], params['pc_max'], params['npc'])
        fig = plot_properties(system, Sw, kro, krw)
        pc_fig = plot_capillary_pressure(system, Sw, pcwo)
        df = pd.DataFrame({
            'Water Saturation (Sw)': Sw,
            'Oil Relative Permeability': kro,
            'Water Relative Permeability': krw,
            'Capillary Pressure (psi)': pcwo
        })
    elif system == 'Gas-Oil System':
        Sg = np.linspace(params['Sgc'], 1 - params['Sl'], 100)
        Sl = 1 - Sg
        kro, krg, pcgo = calculate_properties_gas_oil(Sg, params['Sgc'], params['Sl'], params['Kro_max'], params['krg_max'], params['ng'], params['ngo'], params['pc_max'], params['npg'])
        fig = plot_properties(system, Sl, krg, kro)
        pc_fig = plot_capillary_pressure(system, Sl, pcgo)
        df = pd.DataFrame({
            'Liquid Saturation (Sl)': Sl,
            'Oil Relative Permeability': kro,
            'Gas Relative Permeability': krg,
            'Capillary Pressure (psi)': pcgo
        })
    elif system == 'Gas-Water System':
        Sg = np.linspace(params['Sgc'], 1 - params['Swc'], 100)
        Sw = 1 - Sg
        krg, krw, pcgw = calculate_properties_gas_water(Sg, Sw, params['Sgc'], params['Swc'], params['Krg_max'], params['krw_max'], params['ng'], params['nw'], params['pc_max'], params['npc'])
        fig = plot_properties(system, Sw, krg, krw)
        pc_fig = plot_capillary_pressure(system, Sw, pcgw)
        df = pd.DataFrame({
            'Gas Saturation (Sg)': Sg,
            'Gas Relative Permeability': krg,
            'Water Relative Permeability': krw,
            'Capillary Pressure (psi)': pcgw
        })
    return fig, pc_fig, df

# Streamlit app
st.title('Relative Permeability Curves Generator')

default_params = {
    'Water-Oil System': {'Swc': 0.25, 'Kro_max': 0.85, 'Sorw': 0.35, 'krw_max': 0.4, 'no': 0.9, 'nw': 1.5, 'npc': 0.71, 'pc_max': 20.0},
    'Gas-Oil System': {'Sgc': 0.05, 'Kro_max': 0.60, 'Sl': 0.48, 'krg_max': 0.95, 'ng': 0.6, 'ngo': 1.2, 'npg': 0.51, 'pc_max': 30.0},
    'Gas-Water System': {'Sgc': 0.05, 'Krg_max': 0.90, 'Swc': 0.2, 'krw_max': 0.35, 'ng': 0.8, 'nw': 1.4, 'npc': 0.61, 'pc_max': 20.0}
}

system = st.selectbox('Select System Type', ['Water-Oil System', 'Gas-Oil System', 'Gas-Water System'])
params = default_params[system]

st.markdown("---")

try:
    col1, col2 = st.columns([0.4, 0.6])
    if system == 'Water-Oil System':
        with col1:
            params['Swc'] = st.slider('Swc', 0.0, 1.0, value=params['Swc'], step=0.01)
            params['Kro_max'] = st.slider('Kro_max', 0.0, 1.0, value=params['Kro_max'], step=0.01)
            params['Sorw'] = st.slider('Sorw', 0.0, 1.0, value=params['Sorw'], step=0.01)
            params['krw_max'] = st.slider('krw_max', 0.0, 1.0, value=params['krw_max'], step=0.01)
            params['no'] = st.slider('no', 0.3, 5.0, value=params['no'], step=0.01)
            params['nw'] = st.slider('nw', 0.3, 5.0, value=params['nw'], step=0.01)
            params['npc'] = st.slider('npc', 0.3, 5.0, value=params['npc'], step=0.01)
            params['pc_max'] = st.number_input('pc_max [psi]', min_value=0.0, max_value=1000.0, value=params['pc_max'], step=0.1)
    elif system == 'Gas-Oil System':
        with col1:
            params['Sgc'] = st.slider('Sgc', 0.0, 1.0, value=params['Sgc'], step=0.01)
            params['Kro_max'] = st.slider('Kro_max', 0.0, 1.0, value=params['Kro_max'], step=0.01)
            params['Sl'] = st.slider('Sl', 0.0, 1.0, value=params['Sl'], step=0.01)
            params['krg_max'] = st.slider('krg_max', 0.0, 1.0, value=params['krg_max'], step=0.01)
            params['ngo'] = st.slider('ngo', 0.3, 5.0, value=params['ngo'], step=0.01)
            params['ng'] = st.slider('ng', 0.3, 5.0, value=params['ng'], step=0.01)
            params['npg'] = st.slider('npg', 0.3, 5.0, value=params['npg'], step=0.01)
            params['pc_max'] = st.number_input('pc_max [psi]', min_value=0.0, max_value=1000.0, value=params['pc_max'], step=0.1)
    elif system == 'Gas-Water System':
        with col1:
            params['Sgc'] = st.slider('Sgc', 0.0, 1.0, value=params['Sgc'], step=0.01)
            params['Krg_max'] = st.slider('Krg_max', 0.0, 1.0, value=params['Krg_max'], step=0.01)
            params['Swc'] = st.slider('Swc', 0.0, 1.0, value=params['Swc'], step=0.01)
            params['krw_max'] = st.slider('krw_max', 0.0, 1.0, value=params['krw_max'], step=0.01)
            params['ng'] = st.slider('ng', 0.3, 5.0, value=params['ng'], step=0.01)
            params['nw'] = st.slider('nw', 0.3, 5.0, value=params['nw'], step=0.01)
            params['npc'] = st.slider('npc', 0.3, 5.0, value=params['npc'], step=0.01)
            params['pc_max'] = st.number_input('pc_max [psi]', min_value=0.0, max_value=1000.0, value=params['pc_max'], step=0.1)

    fig, pc_fig, df = generate_plots_and_data(system, params)
    with col2:
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(pc_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error in calculation: {e}")

# Read the existing PDF file
# Google Drive link
drive_link = "https://drive.google.com/file/d/1JQ1GvX-VJXFG4oEeG7fQdw7o2zVde6H_/view?usp=drive_link"

# Extract file ID from the Google Drive link
file_id = drive_link.split('/d/')[1].split('/')[0]

# Create a downloadable link
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Function to check if the content is a PDF
def is_pdf(content):
    return content[:4] == b'%PDF'

# Download the PDF file
try:
    response = requests.get(download_url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    content = response.content
    if not is_pdf(content):
        print("The file is not a PDF or is corrupted.")
        exit(1)
    
    # Read the PDF data
    pdf_data = BytesIO(content)
    reader = PdfReader(pdf_data)
except requests.exceptions.RequestException as e:
    print(f"Failed to download the PDF file: {e}")
    exit(1)

# Custom CSS to set button height
st.markdown(
    """
    <style>
    .download-button {
        height: 50px !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton button {
        height: 50px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# UI layout
st.markdown("---")

try:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Relative_Permeability", index=False)
    buffer.seek(0)
    
    # Centered buttons with equal width
    col1, col2, col3 = st.columns(3, gap='small')
    
    with col1:
        st.download_button(
            label="Download Excel File",
            data=buffer,
            file_name="relative_permeability.xlsx",
            key="excel-download",
            use_container_width=True
        )
        
    ecl_data = export_to_ecl(system, df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3])  # Corrected variable name

    with col2:
        st.download_button(
            label="Download E-100 Sim INC File",
            data=ecl_data.encode('utf-8'),  # Corrected data encoding
            file_name="relative_permeability.INC",
            mime="text/plain",
            key="E-100-download",
            use_container_width=True
        )

    with col3:
        st.download_button(
            label="Equations Used",
            data=pdf_data,
            file_name="Equations Used - Kr and Pc.pdf",
            mime="application/pdf",
            key="pdf-download",
            use_container_width=True
        )
except Exception as e:
    st.error(f"Error in data export: {e}")

# Attribution
st.markdown(
    """
    ---
    Developed by [Shubham B. Patel](https://www.linkedin.com/in/shubham-patel-1045/), R.E., under the guidance of Reservoir Engineering Consultant [Alan Mourgues](https://www.linkedin.com/in/alan-mourgues/), MSc, founder of [CrowdField](https://www.crowdfield.net/).
    
    This work was carried out as a Case Study for the mentoring & collaboration program, part of the value proposition of the CrowdField community. Read the case study [here](https://www.crowdfield.net/blogposts/case-study-collaborating-with-a-young-engineer-to-develop-a-streamlit-relative-permeability-app).
    
    Interested in getting involved or collaborate? Get in touch at hello@crowdfield.net
    
    Visit [CrowdField.net](https://www.crowdfield.net/) to learn more.
    
    """
)
