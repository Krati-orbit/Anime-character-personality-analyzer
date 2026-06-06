// ==========================================================================
// ANIME PERSONALITY ANALYZER - FRONTEND CONTROLLER
// Handles dynamic rendering, smooth transitions, loading spinner sequence,
// Web Audio API sound effects synth, and AJAX communications.
// ==========================================================================

const QUESTIONS = [
    {
        question: "How do you handle a big problem?",
        options: [
            "Confront it directly with passion and energy, refusing to back down!",
            "Analyze it silently, formulate a strategic plan, and execute it calmly.",
            "Treat it as an exciting challenge to test your strength and grow stronger!",
            "Deal with it calmly, focusing on efficiency and getting the job done perfectly."
        ]
    },
    {
        question: "What is your role in a friend group?",
        options: [
            "The energetic lifewire who keeps everyone laughing and motivated.",
            "The quiet, mysterious guardian who protects them from the shadows.",
            "The simple, carefree one who is always ready for adventure and food.",
            "The disciplined, blunt realist who keeps everyone in check and on track."
        ]
    },
    {
        question: "How do you react when someone betrays you?",
        options: [
            "Forgive them and try to win back their friendship, no matter what.",
            "Understand their motives, sacrifice your feelings, and protect the bigger picture.",
            "Fight them to teach them a lesson, but carry no long-term grudges.",
            "Manipulate the situation to deliver your absolute justice."
        ]
    },
    {
        question: "What motivates you the most?",
        options: [
            "The dream of achieving a legendary title and proving myself to everyone.",
            "Protecting my loved ones and bringing peace, even if I must suffer in silence.",
            "Finding strong challenges and pushing my limits to become the strongest version of myself.",
            "Creating a perfect, just world by purging those who act dishonorably."
        ]
    },
    {
        question: "How do you make important decisions?",
        options: [
            "Trust my gut feelings and follow my heart, regardless of the rules or risks.",
            "Calculate every possibility, weigh the strategic outcomes, and decide logically.",
            "Choose the path that offers the most exciting challenge and personal growth.",
            "Make the hard, practical choice based on duty and strict discipline."
        ]
    },
    {
        question: "What is your fighting style? (metaphorically)",
        options: [
            "Raw energy, loud shouts, and fighting with absolute passion and emotion.",
            "Strategic, deceptive, and predicting the opponent's moves steps ahead.",
            "Martial arts mastery, focusing on power, speed, and purely enjoying the battle.",
            "Swift, disciplined precision, leaving zero margin for error."
        ]
    },
    {
        question: "What do people misunderstand about you?",
        options: [
            "They think I'm just loud and silly, but I carry deep pain and care for everyone.",
            "They think I am cold and heartless, but I am actually deeply protective and caring.",
            "They think I'm simple and naive, but I possess an iron will that cannot be broken.",
            "They think I'm just a normal, model citizen, but I hold supreme ambition."
        ]
    },
    {
        question: "Your biggest strength?",
        options: [
            "My unwavering determination—I never, ever give up, no matter the odds.",
            "My mind—high intelligence, strategic plotting, and analytical depth.",
            "My ability to push physical boundaries and adapt in the heat of battle.",
            "My flawless discipline, execution speed, and absolute focus."
        ]
    },
    {
        question: "How do you spend your free time?",
        options: [
            "Hanging out with close friends, sharing a hearty meal, and laughing.",
            "Reflecting in solitude, reading, or planning future goals.",
            "Training hard to sharpen my skills, or exploring challenging new environments.",
            "Cleaning and organizing my space meticulously so that everything is in its place."
        ]
    },
    {
        question: "What is your life goal?",
        options: [
            "To live with complete freedom, embark on adventures, and protect my crew.",
            "To lead, protect my community, and ultimately gain the respect of everyone.",
            "To ensure peace and stability, even if it requires carrying a heavy cross.",
            "To rewrite the system, enforce absolute order, and eliminate corruption."
        ]
    }
];

// ==========================================================================
// WEB AUDIO API - SOUND SYNTHESIS ENGINE
// Generates interface feedback sounds on the fly. No asset loading required.
// ==========================================================================

let audioCtx = null;
let soundEnabled = localStorage.getItem("soundEnabled") !== "false";

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function playSynthSound(freqStart, freqEnd, type, volume, duration) {
    if (!soundEnabled) return;
    try {
        initAudio();
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc.type = type; // 'sine', 'triangle', 'sawtooth', 'square'
        osc.frequency.setValueAtTime(freqStart, audioCtx.currentTime);
        if (freqEnd && freqEnd !== freqStart) {
            osc.frequency.exponentialRampToValueAtTime(freqEnd, audioCtx.currentTime + duration);
        }
        
        gainNode.gain.setValueAtTime(volume, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
        
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.warn("Web Audio API failed to synthesize sound:", e);
    }
}

const soundEffects = {
    hover: () => playSynthSound(900, 1100, 'triangle', 0.015, 0.04),
    select: () => playSynthSound(500, 750, 'sine', 0.08, 0.12),
    transition: () => playSynthSound(200, 480, 'sine', 0.1, 0.5)
};

// DOM Elements
const profileScreen = document.getElementById("profile-screen");
const profileForm = document.getElementById("profile-form");
const usernameInput = document.getElementById("username-input");
const userageInput = document.getElementById("userage-input");
const usergenderInput = document.getElementById("usergender-input");

const landingScreen = document.getElementById("landing-screen");
const quizScreen = document.getElementById("quiz-screen");
const startBtn = document.getElementById("start-btn");
const questionCounter = document.getElementById("question-counter");
const progressBarFill = document.getElementById("progress-bar-fill");
const questionText = document.getElementById("question-text");
const optionsContainer = document.getElementById("options-container");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingTitle = document.getElementById("loading-title");
const loadingStatus = document.getElementById("loading-status");

const soundToggleBtn = document.getElementById("sound-toggle-btn");
const soundIcon = document.getElementById("sound-icon");
const soundStatusText = soundToggleBtn ? soundToggleBtn.querySelector(".sound-status-text") : null;

// App State
let currentQuestionIndex = 0;
const userAnswers = [];

// Audio toggle button functionality
function updateAudioButtonUI() {
    if (soundToggleBtn) {
        if (soundEnabled) {
            soundToggleBtn.classList.remove("muted");
            if (soundIcon) soundIcon.textContent = "🔊";
            if (soundStatusText) soundStatusText.textContent = "SOUNDS ON";
        } else {
            soundToggleBtn.classList.add("muted");
            if (soundIcon) soundIcon.textContent = "🔇";
            if (soundStatusText) soundStatusText.textContent = "SOUNDS MUTED";
        }
    }
}

if (soundToggleBtn) {
    soundToggleBtn.addEventListener("click", () => {
        soundEnabled = !soundEnabled;
        localStorage.setItem("soundEnabled", soundEnabled);
        updateAudioButtonUI();
        if (soundEnabled) {
            soundEffects.select();
        }
    });
}

// Inject welcome greeting in landing screen
function injectPersonalizedGreeting(name) {
    const subtitleEl = landingScreen.querySelector(".subtitle");
    if (subtitleEl) {
        let welcomeText = landingScreen.querySelector(".welcome-user-text");
        if (!welcomeText) {
            welcomeText = document.createElement("div");
            welcomeText.className = "welcome-user-text";
            subtitleEl.parentNode.insertBefore(welcomeText, subtitleEl.nextSibling);
        }
        welcomeText.innerHTML = `Welcome, <strong>${name}</strong>!`;
    }
}

// Check if user profile is already cached in session storage on page load
window.addEventListener("DOMContentLoaded", () => {
    updateAudioButtonUI();
    
    // Bind hover sounds to character hover cards
    const cards = document.querySelectorAll(".char-preview-card");
    cards.forEach(card => {
        card.addEventListener("mouseenter", () => soundEffects.hover());
    });
    
    // Bind hover sound to profile submit button & start button
    const buttons = document.querySelectorAll(".btn, .sound-toggle-btn");
    buttons.forEach(btn => {
        btn.addEventListener("mouseenter", () => soundEffects.hover());
    });

    const cachedName = sessionStorage.getItem("userName");
    const cachedAge = sessionStorage.getItem("userAge");
    const cachedGender = sessionStorage.getItem("userGender");
    
    if (cachedName && cachedAge && cachedGender) {
        if (profileScreen) profileScreen.classList.add("hidden");
        if (landingScreen) {
            landingScreen.classList.remove("hidden");
            injectPersonalizedGreeting(cachedName);
        }
    }
});

// Profile Form Submit Action
if (profileForm) {
    profileForm.addEventListener("submit", (e) => {
        e.preventDefault();
        
        const name = usernameInput.value.trim();
        const age = userageInput.value;
        const gender = usergenderInput.value;
        
        if (!name || !age || !gender) {
            alert("Please fill out all profile fields to proceed!");
            return;
        }
        
        // Save details in browser session storage
        sessionStorage.setItem("userName", name);
        sessionStorage.setItem("userAge", age);
        sessionStorage.setItem("userGender", gender);
        
        // Play transition sound
        soundEffects.transition();
        
        injectPersonalizedGreeting(name);
        
        // Transition screens
        profileScreen.classList.remove("fade-in");
        profileScreen.classList.add("hidden");
        
        landingScreen.classList.remove("hidden");
        landingScreen.classList.add("fade-in");
    });
}

// Start Quiz Action
if (startBtn) {
    startBtn.addEventListener("click", () => {
        soundEffects.transition();
        landingScreen.classList.add("hidden");
        // Remove hidden and trigger fade-in
        quizScreen.classList.remove("hidden");
        quizScreen.classList.add("fade-in");
        renderQuestion();
    });
}

// Render the active question with options
function renderQuestion() {
    if (currentQuestionIndex >= QUESTIONS.length) {
        submitQuiz();
        return;
    }
    
    const currentQuestion = QUESTIONS[currentQuestionIndex];
    
    // Update question metadata headers
    questionCounter.textContent = `Question ${currentQuestionIndex + 1} of ${QUESTIONS.length}`;
    
    // Update progress bar percentage
    const progressPercent = ((currentQuestionIndex + 1) / QUESTIONS.length) * 100;
    progressBarFill.style.width = `${progressPercent}%`;
    
    // Update text
    questionText.textContent = currentQuestion.question;
    
    // Clear and build options list
    optionsContainer.innerHTML = "";
    currentQuestion.options.forEach((optionText, idx) => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.textContent = optionText;
        
        // Audio synthesis hover & click events
        btn.addEventListener("mouseenter", () => soundEffects.hover());
        btn.addEventListener("click", () => {
            soundEffects.select();
            handleOptionSelection(idx, btn);
        });
        
        optionsContainer.appendChild(btn);
    });
}

// Handle clicking of option buttons
function handleOptionSelection(selectedIndex, buttonElement) {
    // Disable all option clicks immediately to prevent double tapping
    const allButtons = optionsContainer.querySelectorAll(".option-btn");
    allButtons.forEach(btn => btn.style.pointerEvents = "none");
    
    // Add highlighting class
    buttonElement.classList.add("selected");
    
    // Save response
    userAnswers.push(selectedIndex);
    
    // Delay transition slightly to let the highlight animation play out beautifully
    setTimeout(() => {
        // Apply fade-out transition classes
        quizScreen.classList.remove("fade-in");
        quizScreen.style.opacity = 0;
        quizScreen.style.transform = "translateY(-10px)";
        
        setTimeout(() => {
            currentQuestionIndex++;
            renderQuestion();
            
            if (currentQuestionIndex < QUESTIONS.length) {
                // Restore styles and trigger fade-in
                quizScreen.style.opacity = "";
                quizScreen.style.transform = "";
                quizScreen.classList.add("fade-in");
            }
        }, 200); // Time of slide out transition
    }, 250); // Time user gets to see highlight
}

// Submit Quiz and triggers loading spinner screen
function submitQuiz() {
    soundEffects.transition();
    
    // Reveal full-screen loading overlay
    loadingOverlay.classList.remove("hidden");
    
    // Dynamic status text sequence
    const statuses = [
        { title: "Chakra Assembly", status: "Tuning your spiritual energy...", delay: 0 },
        { title: "Trait Vector Modeling", status: "Mapping answers to 10-dimensional personality traits...", delay: 500 },
        { title: "KNN Classification", status: "Projecting vector into character clusters (n_neighbors=3)...", delay: 1100 }
    ];
    
    statuses.forEach(step => {
        setTimeout(() => {
            loadingTitle.textContent = step.title;
            loadingStatus.textContent = step.status;
        }, step.delay);
    });
    
    const minLoadingTime = 1600; // Minimum time loading spinner shows (1.6s)
    const startTime = Date.now();
    
    // Submit post request to backend API
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ answers: userAnswers })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || "Server Error"); });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            const elapsed = Date.now() - startTime;
            const remainingDelay = Math.max(0, minLoadingTime - elapsed);
            
            // Wait for remaining loading time to finish before redirecting (avoids visual jarring)
            setTimeout(() => {
                window.location.href = `/result?char=${data.character}&score=${data.score}`;
            }, remainingDelay);
        } else {
            throw new Error(data.error || "Unknown server response.");
        }
    })
    .catch(error => {
        console.error("Submission failed:", error);
        setTimeout(() => {
            loadingOverlay.classList.add("hidden");
            alert(`Analysis failed: ${error.message}\n\nPlease refresh and try again.`);
            resetQuiz();
        }, 500);
    });
}

// Helper to reset quiz state in case of failure
function resetQuiz() {
    currentQuestionIndex = 0;
    userAnswers.length = 0;
    quizScreen.classList.add("hidden");
    landingScreen.classList.remove("hidden");
    landingScreen.classList.add("fade-in");
}
