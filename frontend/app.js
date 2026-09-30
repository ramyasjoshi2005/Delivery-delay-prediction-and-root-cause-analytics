document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Gather all inputs (including hiddens)
    const inputs = document.querySelectorAll('#prediction-form input, #prediction-form select');
    const data = {};
    
    inputs.forEach(input => {
        if(input.tagName === 'BUTTON') return;
        
        let val = input.value;
        // Convert numbers if necessary
        if(input.type === 'number' || input.id === 'order_month' || input.id === 'order_day_of_week' || input.id === 'is_weekend' || input.id === 'quarter' || input.id === 'peak_season') {
            val = Number(val);
        }
        data[input.id] = val;
    });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        document.getElementById('prediction-result').classList.remove('hidden');
        document.getElementById('classification').innerText = result.classification;
        document.getElementById('classification').style.color = result.classification === 'High Risk' ? 'red' : 'green';
        
        document.getElementById('probability').innerText = (result.probability * 100).toFixed(2) + '%';
        document.getElementById('threshold').innerText = result.threshold;
        
        const explanationList = document.getElementById('explanation-list');
        explanationList.innerHTML = '';
        
        result.explanation.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.feature}</strong> ${item.direction} risk (SHAP value: ${item.contribution})`;
            explanationList.appendChild(li);
        });
        
    } catch (err) {
        console.error(err);
        alert('Error making prediction.');
    }
});
