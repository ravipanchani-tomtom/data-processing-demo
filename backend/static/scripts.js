function openTab(evt, tabName) {
    var i, tabcontent, tabbuttons;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tabbuttons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabbuttons.length; i++) {
        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Default open tab
document.addEventListener("DOMContentLoaded", function() {
    document.querySelector(".tab-button").click();
    fetchDatasets();
});

async function fetchDatasets() {
    console.log("Fetching datasets");
    try {
        const response = await fetch("/datasets");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Datasets fetched:", data);
        const datasetSelect = document.getElementById("dataset-select");
        data.datasets.forEach(dataset => {
            const option = document.createElement("option");
            option.value = dataset;
            option.textContent = dataset;
            datasetSelect.appendChild(option);
        });
    } catch (error) {
        console.error("Error fetching datasets:", error);
    }
}

async function fetchSample() {
    const dataset = document.getElementById("dataset-select").value;
    const response = await fetch(`/fetch_sample`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ dataset: dataset })
    });
    const data = await response.json();
    document.getElementById("sample-text").innerText = data.text;
}

async function preprocessText(option) {
    const text = document.getElementById("sample-text").innerText;
    const response = await fetch(`/preprocess`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ dataset: option, text: text })
    });
    const data = await response.json();
    document.getElementById("result").innerText = data.processed_text;
}

async function augmentText(option) {
    const text = document.getElementById("sample-text").innerText;
    const response = await fetch(`/augment`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ dataset: option, text: text })
    });
    const data = await response.json();
    document.getElementById("result").innerText = data.processed_text;
}

function tokenizeText() {
    preprocessText("tokenize");
}

function padText() {
    preprocessText("pad");
}

function embedText() {
    preprocessText("embed");
}

function synonymReplacement() {
    augmentText("synonym_replacement");
}

function randomInsertion() {
    augmentText("random_insertion");
}