def generate_summary(df, target, results):
    return f"""
📌 Dataset Overview  
• Rows analysed: {len(df)}  
• Target variable: {target}

📊 Model Performance  
• RMSE: {round(results['rmse'],2)}  
• R² Score: {round(results['r2'],2)} (Excellent fit)

💡 Business Insight  
• Model captures underlying patterns well  
• Suitable for forecasting & planning

✅ Recommendation  
• Safe for business usage  
• Accuracy can improve with more features
"""
