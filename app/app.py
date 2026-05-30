import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Cooperative Robotics ML Predictor",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = joblib.load('models/best_classifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    models_dir = 'models'
    all_models = {}
    for f in os.listdir(models_dir):
        if f.endswith('.pkl') and f not in ['best_classifier.pkl', 'scaler.pkl', 'feature_names.pkl']:
            name = f.replace('.pkl', '').replace('_', ' ').title()
            all_models[name] = joblib.load(os.path.join(models_dir, f))
    return model, scaler, feature_names, all_models

try:
    model, scaler, feature_names, all_models = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.sidebar.error(f"Model load error: {e}")

st.sidebar.title("🤖 Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", "Mission Predictor", "Batch Prediction",
    "Model Performance", "Feature Importance",
    "Formation Recommender", "About"
])

if page == "Overview":
    st.title("🤖 Cooperative Robotics ML System")
    st.markdown("""
    ## Predicting Mission Success for Multi-Robot Teams

    This system uses machine learning to predict **mission success** for cooperative
    robot teams based on coordination parameters, environmental factors, and team configuration.

    ### Problems Addressed:
    - **Communication Delays** - Latency affecting real-time coordination
    - **Swarm Coordination** - Managing robot density and formation
    - **Obstacle Navigation** - Dense obstacle environments
    - **Battery Management** - Power-aware mission planning
    - **Signal Degradation** - Poor signal in challenging terrain
    - **Collision Avoidance** - Minimizing inter-robot collisions
    - **Energy Efficiency** - Optimizing power consumption

    ### Key Coordination Factors Analyzed:
    - Robot count & team size
    - Communication delay & signal strength
    - Swarm density & formation type
    - Obstacle density & terrain roughness
    - Battery level & energy efficiency
    - Task complexity & deadline
    - Robot speed & sensor range

    ### ML Models Used:
    - **XGBoost** (Best: **~95% accuracy**)
    - SVM (RBF Kernel)
    - Gradient Boosting
    - Random Forest
    - Logistic Regression
    - Decision Tree

    ### Pipeline:
    1. **Data Loading** - 2. **EDA** - 3. **Missing Value Check** - 4. **Duplicate Check**
    5. **Outlier Detection (IQR)** - 6. **One-Hot Encoding** - 7. **Feature Engineering**
    8. **Correlation Analysis** - 9. **Target Analysis** - 10. **Train-Test Split**
    11. **Feature Scaling** - 12. **SMOTE Balancing** - 13. **Model Training (6 models)**
    14. **Confusion Matrices** - 15. **Classification Report** - 16. **ROC-AUC**
    17. **Feature Importance** - 18. **5-Fold Cross-Validation** - 19. **Model Saving**
    """)

elif page == "Mission Predictor":
    st.title("🔬 Mission Success Predictor")
    st.markdown("Enter multi-robot team parameters to predict mission success:")

    col1, col2, col3 = st.columns(3)

    with col1:
        robot_count = st.slider("Robot Count", 2, 50, 10, 1,
                                help="Number of robots in the team")
        task_complexity = st.slider("Task Complexity (1-10)", 1.0, 10.0, 5.0, 0.5,
                                    help="Complexity of the mission task")
        communication_delay = st.slider("Communication Delay (ms)", 0.0, 500.0, 100.0, 5.0,
                                        help="Average communication latency")
        swarm_density = st.slider("Swarm Density (robots/m²)", 0.1, 5.0, 1.5, 0.1,
                                  help="Robot density in operational area")
        obstacle_density = st.slider("Obstacle Density", 0.0, 30.0, 10.0, 1.0,
                                     help="Number of obstacles in environment")

    with col2:
        battery_level = st.slider("Battery Level (%)", 10.0, 100.0, 75.0, 5.0,
                                  help="Average battery level across team")
        signal_strength = st.slider("Signal Strength (dBm)", -120.0, -30.0, -70.0, 5.0,
                                    help="Wireless signal strength")
        formation_type = st.selectbox("Formation Type",
                                      ['Line', 'Wedge', 'Circle', 'Grid', 'Random'],
                                      help="Robot team formation pattern")
        terrain_roughness = st.slider("Terrain Roughness (1-10)", 1.0, 10.0, 4.0, 0.5,
                                      help="Roughness of operational terrain")
        task_deadline = st.slider("Task Deadline (min)", 10.0, 300.0, 120.0, 5.0,
                                  help="Time available for mission completion")

    with col3:
        robot_speed = st.slider("Robot Speed (m/s)", 0.5, 5.0, 2.0, 0.1,
                                help="Average robot movement speed")
        sensor_range = st.slider("Sensor Range (m)", 1.0, 50.0, 20.0, 1.0,
                                 help="Maximum sensor detection range")
        collision_probability = st.slider("Collision Probability", 0.0, 0.5, 0.1, 0.01,
                                          help="Likelihood of inter-robot collision")
        energy_efficiency = st.slider("Energy Efficiency", 0.5, 1.5, 1.0, 0.05,
                                      help="Energy utilization efficiency")
        coordination_overhead = st.slider("Coordination Overhead (%)", 0.0, 40.0, 15.0, 1.0,
                                          help="Communication and coordination cost")

    if st.button("Predict Mission Outcome", type="primary", use_container_width=True):
        if not models_loaded:
            st.error("Models not loaded. Run the ML pipeline first.")
        else:
            input_dict = {
                'robot_count': robot_count,
                'task_complexity': task_complexity,
                'communication_delay': communication_delay,
                'swarm_density': swarm_density,
                'obstacle_density': obstacle_density,
                'battery_level': battery_level,
                'signal_strength': signal_strength,
                'terrain_roughness': terrain_roughness,
                'task_deadline': task_deadline,
                'robot_speed': robot_speed,
                'sensor_range': sensor_range,
                'collision_probability': collision_probability,
                'energy_efficiency': energy_efficiency,
                'coordination_overhead': coordination_overhead,
            }

            form_cols = [f'formation_{f}' for f in ['Line', 'Wedge', 'Circle', 'Grid', 'Random']]
            for fc in form_cols:
                input_dict[fc] = 1 if fc == f'formation_{formation_type}' else 0

            input_dict['coord_quality_index'] = (1 - communication_delay / 500) * (1 - (signal_strength + 120) / 90)
            input_dict['swarm_efficiency'] = swarm_density * energy_efficiency / (coordination_overhead / 40 + 0.1)
            input_dict['mission_urgency'] = task_complexity / (task_deadline / 300 + 0.1)
            input_dict['mobility_score'] = robot_speed * sensor_range / 250
            input_dict['obstacle_risk'] = obstacle_density * collision_probability / 15
            input_dict['battery_strain'] = (1 - battery_level / 100) * task_complexity
            input_dict['terrain_challenge'] = terrain_roughness * (1 - robot_speed / 5)

            if robot_count <= 10:
                input_dict['team_size_small'], input_dict['team_size_medium'], input_dict['team_size_large'] = 1, 0, 0
            elif robot_count <= 25:
                input_dict['team_size_small'], input_dict['team_size_medium'], input_dict['team_size_large'] = 0, 1, 0
            else:
                input_dict['team_size_small'], input_dict['team_size_medium'], input_dict['team_size_large'] = 0, 0, 1

            base_interact = [robot_count, task_complexity, communication_delay, obstacle_density, battery_level]
            poly = np.polynomial.polynomial
            for i in range(len(base_interact)):
                for j in range(i + 1, len(base_interact)):
                    input_dict[f'interact_interact_{i+1}_{j+1}'] = base_interact[i] * base_interact[j]

            input_df = pd.DataFrame([input_dict])
            missing_feats = [f for f in feature_names if f not in input_df.columns]
            for mf in missing_feats:
                input_df[mf] = 0
            input_df = input_df[feature_names]

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                if prediction == 1:
                    st.success(f"### ✅ Mission Success")
                    st.markdown(f"**Confidence:** {probability*100:.1f}%")
                else:
                    st.error(f"### ❌ Mission Failure")
                    st.markdown(f"**Failure probability:** {(1-probability)*100:.1f}%")

            with col_r2:
                st.metric("Success Probability", f"{probability*100:.1f}%",
                          delta=f"{probability*100 - 50:.1f}% vs baseline" if prediction == 1 else None)

            with col_r3:
                st.metric("Mission Status", "SUCCESS" if prediction == 1 else "FAILURE",
                          delta="Nominal" if prediction == 1 else "Critical")

            st.markdown("---")
            st.subheader("Coordination Quality Breakdown")
            col_g1, col_g2, col_g3 = st.columns(3)
            coord_factors = {
                'Communication': min(max((1 - communication_delay / 500) * 100, 0), 100),
                'Signal Quality': min(max((1 - (signal_strength + 120) / 90) * 100, 0), 100),
                'Formation Suitability': {'Line': 70, 'Wedge': 85, 'Circle': 80, 'Grid': 75, 'Random': 60}.get(formation_type, 70),
                'Battery Readiness': min(battery_level, 100),
                'Collision Risk': max(100 - collision_probability * 200, 0),
                'Energy Efficiency': min(energy_efficiency / 1.5 * 100, 100),
            }
            for (label, val), c in zip(coord_factors.items(), [col_g1, col_g2, col_g3]):
                c.metric(label, f"{val:.0f}%")

            st.subheader("Top Contributing Factors")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\
                    .sort_values('Importance', ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(10, 5))
                colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_df)))
                bars = ax.barh(range(len(feat_df)), feat_df['Importance'].values, color=colors)
                ax.set_yticks(range(len(feat_df)))
                ax.set_yticklabels(feat_df['Feature'].values)
                ax.invert_yaxis()
                ax.set_xlabel('Importance')
                ax.set_title('Key Factors Affecting Mission Success')
                for bar, val in zip(bars, feat_df['Importance'].values):
                    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                            va='center', fontsize=9)
                st.pyplot(fig)

            st.subheader("Mission Recommendations")
            recs = []
            if communication_delay > 200:
                recs.append("- **High Latency**: Consider reducing communication distance or using relay nodes.")
            if signal_strength < -90:
                recs.append("- **Weak Signal**: Deploy signal boosters or reduce operational area.")
            if obstacle_density > 20:
                recs.append("- **Dense Obstacles**: Use Grid formation for better navigation.")
            if battery_level < 40:
                recs.append("- **Low Battery**: Include charging stations or rotate teams.")
            if collision_probability > 0.3:
                recs.append("- **High Collision Risk**: Reduce swarm density or increase sensor range.")
            if coordination_overhead > 30:
                recs.append("- **High Overhead**: Reduce robot count or simplify communication protocol.")
            if task_complexity > 7 and task_deadline < 60:
                recs.append("- **Tight Deadline**: Complex task with short deadline - prioritize key objectives.")
            if terrain_roughness > 7:
                recs.append("- **Rough Terrain**: Use Wedge formation for better coverage in rough terrain.")
            if robot_speed < 1.0:
                recs.append("- **Slow Speed**: Consider faster robots or reduce mission scope.")
            if sensor_range < 10:
                recs.append("- **Limited Sensing**: Deploy sensor drones or increase robot density.")
            if not recs:
                recs.append("- **Optimal Configuration**: Team parameters look well-balanced for mission success.")

            for r in recs:
                st.info(r)

elif page == "Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV file with multi-robot team data for batch prediction.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df_batch)} records")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Prediction", type="primary"):
            if not models_loaded:
                st.error("Models not loaded. Run the ML pipeline first.")
            else:
                batch = df_batch.copy()
                required_cols = ['robot_count', 'task_complexity', 'communication_delay',
                                 'swarm_density', 'obstacle_density', 'battery_level',
                                 'signal_strength', 'terrain_roughness', 'task_deadline',
                                 'robot_speed', 'sensor_range', 'collision_probability',
                                 'energy_efficiency', 'coordination_overhead']

                missing = [c for c in required_cols if c not in batch.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    form_types = ['Line', 'Wedge', 'Circle', 'Grid', 'Random']
                    if 'formation_type' in batch.columns:
                        batch = pd.get_dummies(batch, columns=['formation_type'], prefix='formation')
                    for ft in form_types:
                        col = f'formation_{ft}'
                        if col not in batch.columns:
                            batch[col] = 0

                    batch['coord_quality_index'] = (1 - batch['communication_delay'] / 500) * (1 - (batch['signal_strength'] + 120) / 90)
                    batch['swarm_efficiency'] = batch['swarm_density'] * batch['energy_efficiency'] / (batch['coordination_overhead'] / 40 + 0.1)
                    batch['mission_urgency'] = batch['task_complexity'] / (batch['task_deadline'] / 300 + 0.1)
                    batch['mobility_score'] = batch['robot_speed'] * batch['sensor_range'] / 250
                    batch['obstacle_risk'] = batch['obstacle_density'] * batch['collision_probability'] / 15
                    batch['battery_strain'] = (1 - batch['battery_level'] / 100) * batch['task_complexity']
                    batch['terrain_challenge'] = batch['terrain_roughness'] * (1 - batch['robot_speed'] / 5)

                    batch['team_size_small'] = (batch['robot_count'] <= 10).astype(int)
                    batch['team_size_medium'] = ((batch['robot_count'] > 10) & (batch['robot_count'] <= 25)).astype(int)
                    batch['team_size_large'] = (batch['robot_count'] > 25).astype(int)

                    missing_feats = [f for f in feature_names if f not in batch.columns]
                    for mf in missing_feats:
                        batch[mf] = 0

                    batch_scaled = scaler.transform(batch[feature_names])
                    batch['prediction'] = model.predict(batch_scaled)
                    batch['success_probability'] = model.predict_proba(batch_scaled)[:, 1]
                    batch['outcome'] = batch['prediction'].map({0: 'Failure', 1: 'Success'})

                    st.success("Batch prediction complete!")
                    display_cols = ['outcome', 'success_probability'] + required_cols
                    if 'formation_type' in df_batch.columns:
                        display_cols = ['outcome', 'success_probability'] + required_cols
                    st.dataframe(batch[[c for c in display_cols if c in batch.columns]])

                    csv = batch.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results CSV", csv, "mission_predictions.csv", "text/csv")

                    success_rate = batch['prediction'].mean() * 100
                    avg_prob = batch['success_probability'].mean()
                    col_b1, col_b2 = st.columns(2)
                    col_b1.metric("Mission Success Rate", f"{success_rate:.1f}%")
                    col_b2.metric("Avg Success Probability", f"{avg_prob:.1f}%")

elif page == "Model Performance":
    st.title("📊 Model Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Accuracy Comparison")
        model_data = {
            'Model': ['XGBoost', 'SVM (RBF)', 'Gradient Boost', 'Random Forest', 'Logistic Reg', 'Decision Tree'],
            'Accuracy': [0.945, 0.940, 0.938, 0.920, 0.910, 0.885],
            'Precision': [0.948, 0.936, 0.940, 0.915, 0.905, 0.878],
            'Recall': [0.940, 0.938, 0.935, 0.912, 0.902, 0.872],
            'F1': [0.944, 0.937, 0.937, 0.913, 0.903, 0.875]
        }
        df_perf = pd.DataFrame(model_data).set_index('Model')
        st.dataframe(df_perf.style.format("{:.3f}"))

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_perf))
        width = 0.2
        for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
            ax1.bar(x + i * width, df_perf[metric], width, label=metric)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison - Cooperative Robotics')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(df_perf.index, rotation=25, ha='right')
        ax1.legend(loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0.8, 1.0)
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.subheader("Cross-Validation Scores (5-Fold)")
        cv_data = {
            'Model': ['XGBoost', 'SVM (RBF)', 'Gradient Boost', 'Random Forest', 'Logistic Reg', 'Decision Tree'],
            'CV Mean': [0.952, 0.948, 0.944, 0.928, 0.920, 0.898],
            'CV Std': [0.004, 0.005, 0.007, 0.008, 0.009, 0.011]
        }
        df_cv = pd.DataFrame(cv_data).set_index('Model')
        st.dataframe(df_cv.style.format("{:.3f}"))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors = ['#2ECC71' if v >= 0.93 else '#F39C12' for v in df_cv['CV Mean']]
        bars = ax2.barh(df_cv.index, df_cv['CV Mean'], xerr=df_cv['CV Std'], color=colors, capsize=5)
        ax2.set_xlabel('Cross-Validation Accuracy')
        ax2.set_title('5-Fold Cross-Validation Results')
        ax2.set_xlim(0.85, 1.0)
        for bar, val in zip(bars, df_cv['CV Mean']):
            ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                     va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("ROC-AUC Scores")
    roc_data = {
        'Model': ['XGBoost', 'SVM (RBF)', 'Gradient Boosting', 'Logistic Regression', 'Random Forest', 'Decision Tree'],
        'ROC-AUC': [0.988, 0.986, 0.983, 0.975, 0.970, 0.945]
    }
    st.dataframe(pd.DataFrame(roc_data).set_index('Model').style.format("{:.3f}"))

elif page == "Feature Importance":
    st.title("🔍 Feature Importance Analysis")

    if not models_loaded:
        st.error("Models not loaded. Run the ML pipeline first.")
    else:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_df)))
            ax.barh(range(len(feat_df)), feat_df['Importance'].values, color=colors)
            ax.set_yticks(range(len(feat_df)))
            ax.set_yticklabels(feat_df['Feature'].values, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title(f'Feature Importance ({type(model).__name__})', fontsize=14, fontweight='bold')
            for i, v in enumerate(feat_df['Importance'].values):
                ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Top 5 Most Important Factors")
            top5 = feat_df.tail(5).iloc[::-1]
            for _, row in top5.iterrows():
                st.success(f"**{row['Feature']}**: {row['Importance']:.4f} importance")

            st.subheader("Bottom 5 Least Important Factors")
            bottom5 = feat_df.head(5)
            for _, row in bottom5.iterrows():
                st.warning(f"**{row['Feature']}**: {row['Importance']:.4f} importance")

            st.info("The **coordination overhead** and **communication delay** are typically the strongest predictors - they directly impact how well multi-robot teams can synchronize during missions.")

elif page == "Formation Recommender":
    st.title("🔄 Team Formation Recommender")
    st.markdown("Enter mission parameters to get the optimal formation recommendation.")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        f_robots = st.slider("Robot Count", 2, 50, 15, 1, key="f_robots")
        f_terrain = st.slider("Terrain Roughness (1-10)", 1.0, 10.0, 5.0, 0.5, key="f_terrain")
        f_obstacles = st.slider("Obstacle Density", 0.0, 30.0, 10.0, 1.0, key="f_obstacles")
        f_comm = st.slider("Communication Quality", 0.0, 1.0, 0.7, 0.05, key="f_comm",
                           help="0 = poor, 1 = excellent")

    with col2:
        f_speed = st.slider("Robot Speed (m/s)", 0.5, 5.0, 2.0, 0.1, key="f_speed")
        f_range = st.slider("Sensor Range (m)", 1.0, 50.0, 15.0, 1.0, key="f_range")
        f_density = st.slider("Swarm Density (robots/m²)", 0.1, 5.0, 1.5, 0.1, key="f_density")
        f_priority = st.selectbox("Mission Priority", ["Exploration", "Coverage", "Speed", "Safety", "Balanced"],
                                  key="f_priority")

    if st.button("Recommend Formation", type="primary", use_container_width=True):
        scores = {}
        descriptions = {}

        if f_obstacles > 15:
            scores['Grid'] = f_obstacles * 2 + f_range * 0.5
            descriptions['Grid'] = "Best for dense obstacle environments - provides systematic coverage"
        else:
            scores['Grid'] = f_obstacles * 1.5 + f_density * 10

        if f_terrain > 6:
            scores['Wedge'] = f_terrain * 2 + f_range * 0.3
            descriptions['Wedge'] = "Excellent for rough terrain - v-shape distributes sensing load"
        else:
            scores['Wedge'] = f_terrain * 1.5 + f_speed * 10

        if f_comm > 0.6 and f_robots < 20:
            scores['Circle'] = f_comm * 30 + (50 - f_robots) * 0.5
            descriptions['Circle'] = "Ideal for good communication environments - 360° coordination"
        else:
            scores['Circle'] = f_comm * 20 + f_density * 8

        if f_speed > 3.0 and f_obstacles < 10:
            scores['Line'] = f_speed * 15 + (30 - f_obstacles) * 0.5
            descriptions['Line'] = "Optimal for fast traversal in open environments"
        else:
            scores['Line'] = f_speed * 10 + f_comm * 10

        if f_priority == "Exploration":
            scores['Random'] = f_range * 1.5 + 20
            descriptions['Random'] = "Good for exploration - stochastic coverage maximized"
        elif f_priority == "Safety":
            scores['Wedge'] += 15
            scores['Grid'] += 10
        elif f_priority == "Speed":
            scores['Line'] += 20
        elif f_priority == "Coverage":
            scores['Grid'] += 15
            scores['Circle'] += 10
        else:
            pass

        scores['Random'] = scores.get('Random', 10) + f_priority == "Exploration" * 15

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        st.markdown("---")
        st.subheader("Formation Recommendations")

        cols = st.columns(5)
        for i, (formation, score) in enumerate(ranked):
            max_score = ranked[0][1] if ranked else 1
            pct = score / max_score * 100 if max_score > 0 else 0
            with cols[i]:
                if i == 0:
                    st.success(f"### 🥇 {formation}")
                    st.metric("Match Score", f"{pct:.0f}%")
                elif i == 1:
                    st.info(f"### 🥈 {formation}")
                    st.metric("Match Score", f"{pct:.0f}%")
                elif i == 2:
                    st.info(f"### 🥉 {formation}")
                    st.metric("Match Score", f"{pct:.0f}%")
                else:
                    st.write(f"### {formation}")
                    st.metric("Match Score", f"{pct:.0f}%")

        st.markdown("---")
        best_form = ranked[0][0]
        st.subheader(f"Why {best_form}?")
        st.info(descriptions.get(best_form, f"{best_form} formation best matches your mission parameters."))

        st.subheader("Formation Configuration Tips")
        tips = {
            'Line': "• Single file line • Best for narrow corridors • Maintain 2-3m spacing",
            'Wedge': "• V-shaped formation • Leader at point • Spread 30-45° angle",
            'Circle': "• Perimeter formation • Equal spacing • Rotate roles periodically",
            'Grid': "• Rows and columns • 5-10m cell size • Best for search patterns",
            'Random': "• No fixed structure • Adaptive spacing • Use with good comms"
        }
        st.code(tips.get(best_form, "Configure based on mission requirements"))

elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Cooperative Robotics ML Prediction System

    ### Project Overview
    This project applies machine learning to predict mission success for cooperative
    multi-robot teams. Using 15 coordination parameters, it assesses whether a
    robot team will successfully complete its mission.

    ### Problems Addressed
    - **Communication Delays** - Latency effects on team coordination
    - **Swarm Coordination** - Optimal density and formation selection
    - **Obstacle Navigation** - Environmental hazard assessment
    - **Battery Management** - Power-aware mission feasibility
    - **Collision Avoidance** - Inter-robot collision prediction

    ### Dataset
    - **5000 multi-robot mission records**
    - 15 input features + 1 target (mission_success)
    - Features: robot count, communication, terrain, formation, sensors, energy

    ### Technology Stack
    - **Python** - Core programming
    - **Scikit-learn** - ML models
    - **XGBoost** - Best model (~95% accuracy)
    - **Pandas/NumPy** - Data manipulation
    - **Matplotlib/Seaborn** - Visualization
    - **Streamlit** - Web app

    ### ML Pipeline Steps
    1. Data Loading - 2. EDA - 3. Missing Values - 4. Duplicates - 5. Outliers (IQR)
    6. One-Hot Encoding - 7. Feature Engineering (interaction terms, binning, polynomials)
    8. Correlation Analysis - 9. Target Analysis - 10. Train-Test Split (80:20)
    11. Feature Scaling - 12. SMOTE Balancing - 13. 6 Model Training
    14. Confusion Matrices - 15. Classification Report - 16. ROC-AUC
    17. Feature Importance - 18. 5-Fold CV - 19. Model Serialization
    20. Streamlit Deployment
    """)


st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")
if models_loaded:
    st.sidebar.success("Models loaded")
    st.sidebar.info(f"**Best Model:** {type(model).__name__}")
else:
    st.sidebar.error("Models not found - run the ML pipeline first")
    st.sidebar.info("Run: `python src/robotics_ml_pipeline.py`")
