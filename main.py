import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ==========================================
# 1. AUTO-CREATE DATASET (Ensures code works everywhere)
# ==========================================
def create_dataset_if_missing():
	filename = 'air_quality_global.csv'
	if not os.path.exists(filename):
		print(f"[INFO] '{filename}' not found. Creating extended dataset...")
		# Create a dataset with multiple cities to support the "Prediction" feature
		csv_data = """city,country,latitude,longitude,year,month,pm25_ugm3,no2_ugm3,data_quality,measurement_method,data_source
New York,USA,40.7128,-74.006,2024,1,18.11,35.98,Good,Ref,EPA
New York,USA,40.7128,-74.006,2024,2,27.79,17.71,Good,Ref,EPA
New York,USA,40.7128,-74.006,2024,3,12.05,40.99,Good,Ref,EPA
New York,USA,40.7128,-74.006,2024,4,35.25,17.18,Poor,Ref,EPA
New York,USA,40.7128,-74.006,2024,5,38.39,25.07,Good,Ref,EPA
London,UK,51.5074,-0.1278,2024,1,14.20,25.50,Good,Ref,DEFRA
London,UK,51.5074,-0.1278,2024,2,16.50,28.10,Good,Ref,DEFRA
London,UK,51.5074,-0.1278,2024,3,12.10,22.40,Good,Ref,DEFRA
Tokyo,Japan,35.6762,139.6503,2024,1,15.20,30.50,Good,Ref,JMO
Tokyo,Japan,35.6762,139.6503,2024,2,18.50,32.10,Good,Ref,JMO
Mumbai,India,19.0760,72.8777,2024,1,45.20,50.50,Poor,Ref,CPCB
Mumbai,India,19.0760,72.8777,2024,2,52.10,55.10,Poor,Ref,CPCB
Paris,France,48.8566,2.3522,2024,1,13.20,20.50,Good,Ref,EEA
Lagos,Nigeria,6.5244,3.3792,2024,1,65.20,45.50,Poor,Ref,WHO"""
		with open(filename, "w") as f:
			f.write(csv_data)
		print("[SUCCESS] Dataset created.")


# ==========================================
# 2. HYDROLOGICAL MODELING & NUMERICAL METHODS
# ==========================================
def process_and_train_model():
	df = pd.read_csv('air_quality_global.csv')

	# We use 'New York' data to calibrate our standard Unit Hydrograph model
	model_df = df[df['city'] == 'New York'].copy().reset_index(drop=True)

	# Ensure we have 24 hours of data points for the hydrograph
	if len(model_df) < 24:
		# Repeat data to fill 24 hours if dataset is small
		model_df = pd.concat([model_df] * (24 // len(model_df) + 1), ignore_index=True)
	model_df = model_df.head(24)

	# --- SYNTHESIZE HYDROLOGICAL DATA ---
	model_df['Time_Hour'] = np.arange(24)
	# Rainfall Hyetograph (Synthetic Storm Event)
	rainfall_pattern = [0, 0, 0, 2, 5, 10, 25, 40, 20, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	model_df['Rainfall_mm'] = rainfall_pattern

	# Calculate Rational Method 'C' (Runoff Coefficient) using PM2.5 as proxy for Urbanization
	# Logic: Higher Pollution (PM2.5) -> More Concrete/Traffic -> Higher Runoff
	pm_min, pm_max = model_df['pm25_ugm3'].min(), model_df['pm25_ugm3'].max()
	model_df['Rational_C'] = 0.3 + 0.6 * ((model_df['pm25_ugm3'] - pm_min) / (pm_max - pm_min + 1e-6))

	# Calculate Infiltration (Horton's Equation)
	f0, fc, k = 10, 2, 0.5
	model_df['Infiltration_mmhr'] = fc + (f0 - fc) * np.exp(-k * model_df['Time_Hour'])

	# Calculate Observed Flow (Ground Truth)
	# Flow = (C * Rainfall) - Infiltration
	flow_raw = (model_df['Rational_C'] * model_df['Rainfall_mm']) - model_df['Infiltration_mmhr']
	model_df['Observed_Flow'] = np.maximum(flow_raw, 0)  # Physics: Flow >= 0

	# --- NUMERICAL COMPUTING: MATRIX INVERSION ---
	# Model: Flow = Beta0 + Beta1*Rainfall + Beta2*PM2.5
	# Construct Design Matrix X
	N = len(model_df)
	ones = np.ones(N)
	X = np.column_stack((ones, model_df['Rainfall_mm'].values, model_df['pm25_ugm3'].values))
	y = model_df['Observed_Flow'].values

	# Normal Equation: Beta = (X^T * X)^-1 * X^T * y
	XtX = X.T @ X

	try:
		XtX_inv = np.linalg.inv(XtX)
	except np.linalg.LinAlgError:
		XtX_inv = np.linalg.pinv(XtX)  # Pseudo-inverse

	beta = XtX_inv @ (X.T @ y)

	# Simulate Flow
	flow_simulated = X @ beta
	flow_simulated = np.maximum(flow_simulated, 0)

	return model_df, X, XtX, beta, flow_simulated, y, df


# ==========================================
# 3. OUTPUT GENERATION (Tables & Graphs)
# ==========================================
def show_results(df, XtX, beta, flow_sim, y):
	# --- PRINT TABLES ---
	print("\n" + "=" * 40)
	print("      PROJECT TABLES (Copy to Report)")
	print("=" * 40)

	print("\n[Table 1] Infiltration Rates (First 5 hours)")
	print(df[['Time_Hour', 'Infiltration_mmhr']].head().to_string(index=False))

	print("\n[Table 2] Rational Method Coefficients (Derived from PM2.5)")
	print(df[['Time_Hour', 'pm25_ugm3', 'Rational_C']].head().to_string(index=False))

	print("\n[Table 3] Normal Equation Matrix (XtX) - Matrix Inversion Core")
	print(pd.DataFrame(XtX, columns=['Intercept', 'Rainfall', 'PM2.5'], index=['Intercept', 'Rainfall', 'PM2.5']))

	residuals = y - flow_sim
	mse = np.mean(residuals ** 2)
	print("\n[Table 4] Mean Squared Error")
	print(f"MSE: {mse:.4f} | RMSE: {np.sqrt(mse):.4f}")

	print("\n[Table 5] Peak Flow Comparison")
	print(f"Observed Peak: {y.max():.2f} m3/s")
	print(f"Simulated Peak: {flow_sim.max():.2f} m3/s")

	# --- PLOT FIGURES ---
	print("\n[INFO] Generating Figures... (Close the plot window to continue)")
	fig, axs = plt.subplots(3, 2, figsize=(14, 18))
	plt.subplots_adjust(hspace=0.4)

	# Fig 1: Hyetograph
	axs[0, 0].bar(df['Time_Hour'], df['Rainfall_mm'], color='blue', alpha=0.6)
	axs[0, 0].set_title('Fig 1: Rainfall Hyetograph')
	axs[0, 0].invert_yaxis()
	axs[0, 0].set_ylabel('Rainfall (mm)')

	# Fig 2: Unit Hydrograph
	uh = flow_sim / (df['Rainfall_mm'].sum() + 1e-6)
	axs[0, 1].plot(df['Time_Hour'], uh, 'g-o')
	axs[0, 1].set_title('Fig 2: Unit Hydrograph Curve')

	# Fig 3: Observed vs Simulated
	axs[1, 0].plot(df['Time_Hour'], y, 'b-', label='Observed')
	axs[1, 0].plot(df['Time_Hour'], flow_sim, 'r--', label='Simulated (Regression)')
	axs[1, 0].legend()
	axs[1, 0].set_title('Fig 3: Observed vs Simulated Flow')

	# Fig 4: Residuals
	axs[1, 1].scatter(flow_sim, residuals, color='purple')
	axs[1, 1].axhline(0, color='k', linestyle='--')
	axs[1, 1].set_title('Fig 4: Residual Scatter Plot')

	# Fig 5: Area-Velocity
	vel = 0.5 * (y ** 0.4)
	area = y / (vel + 1e-6)
	axs[2, 0].plot(area, vel, 'k-', lw=2)
	axs[2, 0].set_title('Fig 5: Area-Velocity Relationship')

	axs[2, 1].axis('off')

	plt.savefig('Project_Figures.png')
	plt.show()  # Blocks execution until closed


# ==========================================
# 4. PREDICTION MODULE (Map + Dashboard)
# ==========================================
def run_prediction_dashboard(full_df, beta):
	print("\n" + "=" * 50)
	print("   PREDICT FUTURE RAINFALL & FLOOD RISK")
	print("=" * 50)
	print("Available Cities in Data: New York, London, Tokyo, Mumbai, Paris, Lagos")

	city = input("Enter City Name for Prediction: ").strip()
	if not city: city = "New York"

	# Get city specific data
	city_data = full_df[full_df['city'].str.lower() == city.lower()]

	if city_data.empty:
		print(f"City '{city}' not in historical data. Using Global Average.")
		avg_pm25 = full_df['pm25_ugm3'].mean()
	else:
		avg_pm25 = city_data['pm25_ugm3'].mean()

	# Predict for a hypothetical 50mm storm
	rain_input = 50.0
	# y = b0 + b1*Rain + b2*PM2.5
	pred_flow = beta[0] + beta[1] * rain_input + beta[2] * avg_pm25
	pred_flow = max(pred_flow, 0)

	# Determine Risk
	risk_pct = min((pred_flow / 40.0) * 100, 100)
	risk_level = "LOW"
	if risk_pct > 40: risk_level = "MODERATE"
	if risk_pct > 75: risk_level = "HIGH (FLOOD WARNING)"

	print(f"\n--- PREDICTION FOR {city.upper()} ---")
	print(f"Predicted Runoff: {pred_flow:.2f} m3/s")
	print(f"Risk Level: {risk_level}")

	# Generate HTML Dashboard
	html = f"""
    <html>
    <head>
        <title>Hydrological Forecast: {city}</title>
        <style>
            body {{ display: flex; font-family: Arial, sans-serif; margin: 0; }}
            #sidebar {{ width: 300px; background: #222; color: #fff; padding: 20px; }}
            #map {{ flex: 1; height: 100vh; }}
            .metric {{ margin-bottom: 20px; background: #333; padding: 10px; border-radius: 5px; }}
            .val {{ font-size: 24px; font-weight: bold; color: #00d2ff; }}
            .risk {{ font-size: 20px; font-weight: bold; color: {'#ff4444' if risk_pct > 50 else '#00ff00'}; }}
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Flood Prediction</h2>
            <p>Target: <strong>{city.upper()}</strong></p>
            <div class="metric">
                <label>Avg Urban Density (PM2.5)</label>
                <div class="val">{avg_pm25:.1f}</div>
            </div>
            <div class="metric">
                <label>Predicted Runoff Flow</label>
                <div class="val">{pred_flow:.2f} m3/s</div>
            </div>
            <div class="metric">
                <label>Flood Risk</label>
                <div class="risk">{risk_level}</div>
                <div style="width: 100%; background: #555; height: 10px; margin-top:5px;">
                    <div style="width: {risk_pct}%; background: {'#ff4444' if risk_pct > 50 else '#00ff00'}; height: 100%;"></div>
                </div>
            </div>
            <p><em>Model: Matrix Inversion Regression</em></p>
        </div>
        <iframe id="map" frameborder="0" style="border:0"
            src="https://maps.google.com/maps?q={city}&t=&z=12&ie=UTF8&iwloc=&output=embed">
        </iframe>
    </body>
    </html>
    """

	with open("dashboard.html", "w") as f:
		f.write(html)

	print("[INFO] Dashboard generated. Opening in browser...")
	webbrowser.open("dashboard.html")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
	create_dataset_if_missing()

	# 1. Run Analysis & Show Graphs
	model_df, X, XtX, beta, flow_sim, y, full_df = process_and_train_model()
	show_results(model_df, XtX, beta, flow_sim, y)

	# 2. Run Prediction (After graphs are closed)
	run_prediction_dashboard(full_df, beta)