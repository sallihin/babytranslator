// Speech Recognition (CAA1C15)
// Muhammad Putra Nursallihin
// Student ID: 2082050B

let recognizer;

// One frame is ~23ms of audio.
const NUM_FRAMES = 43;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];

// Initialise Text To Speech
const synth = window.speechSynthesis;

function listen() {
  if (recognizer.isListening()) {
    // If the app is listening, stop listening and pulsing
    recognizer.stopListening();
    document.getElementById("button").classList.remove("pulse-animate");
    return;
  }
  document.getElementById("button").classList.add("pulse-animate");
  recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {

    const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
    const probs = model.predict(input);
    const predLabel = probs.argMax(1).dataSync();

    // Labels ref to index from metadata.json
    let result;
    if (predLabel == 0) {
      result = "I'm uncomfortable";
    } else if (predLabel == 1) {
      result = "I'm hungry";
    } else {
      result = "I'm tired";
    }

    // Display results 
    document.getElementById('console').innerHTML = result + '<img id="tts" src="speech.png">';
    
    // Activate text to speech when clicked
    document.getElementById('console').addEventListener('click', speak); 
    tf.dispose([input, probs, predLabel]);
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true,
    probabilityThreshold: 0.5
  });

  // Stops listening after 4.5seconds
  setTimeout(listen, 4500);
}

function speak() {
  if (synth.speaking) {
    console.error('speechSynthesis.speaking');
    return;
  }
  if (document.getElementById('console').textContent !== '') {
    var utterThis = new SpeechSynthesisUtterance(document.getElementById('console').textContent);
    utterThis.onend = function (event) {
      console.log('SpeechSynthesisUtterance.onend');
    }
    utterThis.onerror = function (event) {
      console.error('SpeechSynthesisUtterance.onerror');
    }
    utterThis.lang = 'en';
    utterThis.rate = 0.8;
    // utterThis.voice = window.speechSynthesis.getVoices()[1];
    synth.speak(utterThis);
  }
}

function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

// Load trained model 
async function app() {
  model = await tf.loadLayersModel('model.json');
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
}

app();