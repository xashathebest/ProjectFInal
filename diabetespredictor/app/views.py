from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponseBadRequest
import numpy as np
import pickle

@csrf_protect
def predict(request):
    features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
    context = {'fields': features}
    
    if request.method == 'POST':
        try:
            # Validate form data
            input_data = []
            for feature in features:
                value = request.POST.get(feature)
                if value is None or value.strip() == '':
                    raise ValueError(f"{feature} is required")
                try:
                    input_data.append(float(value))
                except ValueError:
                    raise ValueError(f"{feature} must be a valid number")

            # Load model and scaler
            try:
                with open('models/diabetes_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('models/scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
            except FileNotFoundError:
                raise ValueError("Model files not found. Please contact administrator.")

            # Prepare and predict
            input_array = np.array(input_data).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1] * 100

            context.update({
                'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                'probability': f'{probability:.2f}%',
                'input_data': dict(zip(features, input_data))
            })
        except ValueError as e:
            context.update({
                'error': str(e)
            })
        except Exception as e:
            context.update({
                'error': f'An unexpected error occurred: {str(e)}'
            })

    return render(request, 'form.html', context)
