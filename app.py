import streamlit as st
import pickle
import time
import random
import plotly.graph_objects as go
import pandas as pd

# Load the machine learning model
model_up = pickle.load(open('fraud_model.pkl', 'rb'))

# Function to predict fraud
def predict_fraud(transaction_type, amount, old_balance):
    # Map transaction_type to numeric values
    transaction_type_mapping = {'Transfer': 0, 'Payment': 1, 'Debit': 2, 'Cash_In': 3, 'Cash_Out': 4}
    numeric_transaction_type = transaction_type_mapping.get(transaction_type, -1)  # -1 if not found
    
    if numeric_transaction_type == -1:
        st.warning("Invalid transaction type. Please select a valid transaction type.")
        return None
    
    # Automatically calculate the new balance
    new_balance = old_balance - amount
    
    # Use the calculated new balance for prediction
    result = model_up.predict_proba([[numeric_transaction_type, amount, old_balance, new_balance]])[:, 1][0]
    return result

# Function to simulate new transaction data
def simulate_new_transaction():
    transaction_type = random.choice(['Transfer', 'Payment', 'Debit', 'Cash_In', 'Cash_Out'])
    amount = round(random.uniform(10, 1000), 2)
    old_balance = round(random.uniform(1000, 10000), 2)
    new_balance = old_balance - amount
    return transaction_type, amount, old_balance, new_balance

# Streamlit App
def main():
    st.title("Real-Time Fraud Detection")

    # Transaction inputs
    amount = st.number_input("Enter transaction amount")
    old_bal = st.number_input("Enter old balance")

    # Radio button for transaction type
    transaction_type = st.radio(
        "Select transaction type:",
        ('Transfer', 'Payment', 'Debit', 'Cash_In', 'Cash_Out')
    )

    # Create an empty list for transaction data
    transaction_data = []

    # Real-time updates section
    st.subheader("Real-Time Updates")

    # Create an empty Plotly figure
    fig = go.Figure()

    # Prediction button
    if st.button('Predict Fraud'):
        predicted_probability = predict_fraud(transaction_type, amount, old_bal)

        if predicted_probability is not None:
            # Display transaction result
            if predicted_probability > 0.5:
                st.warning("This transaction is likely a Fraud with probability {:.2f}%".format(predicted_probability * 100))
                st.error("ALERT: Potential Fraud Detected!")
            else:
                st.success("This transaction is likely Not a Fraud with probability {:.2f}%".format((1 - predicted_probability) * 100))
            
            # Add transaction to history
            transaction_data.append({
                "Transaction Type": transaction_type,
                "Amount": amount,
                "Old Balance": old_bal,
                "New Balance": old_bal - amount,
                "Predicted Probability": predicted_probability
            })

    if st.button('Start Real-Time Monitoring'):
        try:
            while True:
                # Simulate new transaction
                new_transaction_type, new_amount, new_old_balance, new_new_balance = simulate_new_transaction()
                predicted_probability = predict_fraud(new_transaction_type, new_amount, new_old_balance)  # Use new_transaction_type directly

                if predicted_probability is not None:
                    # Display transaction result
                    if predicted_probability > 0.5:
                        st.warning(f"Potential Fraud Detected! Transaction Type: {new_transaction_type}, Amount: {new_amount}, Old Balance: {new_old_balance}, New Balance: {new_new_balance}")
                    else:
                        st.success(f"Transaction Type: {new_transaction_type}, Amount: {new_amount}, Old Balance: {new_old_balance}, New Balance: {new_new_balance}")

                    # Append data to the list
                    transaction_data.append({
                        "Transaction Type": new_transaction_type,
                        "Amount": new_amount,
                        "Old Balance": new_old_balance,
                        "New Balance": new_new_balance,
                        "Predicted Probability": predicted_probability
                    })

                    # Convert the list to a DataFrame
                    transaction_df = pd.DataFrame(transaction_data)

                    # Update the Plotly figure with new data
                    fig.add_trace(go.Scatter(x=transaction_df.index, y=transaction_df["Predicted Probability"], mode='lines+markers', name='Predicted Probability'))
                    
                    # Update the Plotly layout
                    fig.update_layout(title="Real-Time Predicted Probability",
                                      xaxis_title="Transaction",
                                      yaxis_title="Predicted Probability")
                    
                    # Display the updated Plotly figure
                    st.plotly_chart(fig)

                time.sleep(2)  # Simulate a delay between transactions

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    if len(transaction_data) > 0:
        st.subheader("Fraud History")
        history_df = pd.DataFrame(transaction_data)
        st.dataframe(history_df)
        st.plotly_chart(go.Figure(data=[
            go.Scatter(x=history_df.index, y=history_df["Predicted Probability"], mode='lines+markers', name='Predicted Probability')
        ]))

        if st.checkbox("Show transaction details"):
            selected_transaction = st.selectbox("Select a transaction index to view details", history_df.index)
            st.write(history_df.loc[selected_transaction])

if __name__ == '__main__':
    main()
