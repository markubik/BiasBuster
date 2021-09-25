import type { Message } from "./content_script";

export type HateSpeechType = "NORMAL" | "OFFENSIVE" | "HATESPEECH"
export type HyperpartisanType = "NORMAL" | "HYPERPARTISAN"
export type StanceType = "UNRELATED" | "AGREE" | "DISAGREE" | "DISCUSS"

export interface BasicPrediction {
    error: string;
    name: string;
}

export interface Bias {
    bias: 'BALANCED' | 'TAKING_STANCE' | 'MISLEADING' | 'BIASED' | 'STRONGLY_BIASED'
    predictions: {
        hatespeech: BasicPrediction & { prediction: HateSpeechType },
        hyperpartisan: BasicPrediction & { prediction: HyperpartisanType },
        stance: BasicPrediction & { prediction: StanceType }
    }
}

const results = {};

chrome.tabs.onUpdated.addListener((tabId, _, tab) => {
    chrome.runtime.onMessage.addListener(
        function messageListener(message: Message) {
            if (message.type === 'PAGE_INITIALIZED') {
                handlePageInitialized(tabId);
            }
            else if (message.type === 'POPUP_INITIALIZED') {
                handlePopupInitialized(message.url);
            }
        })
    const result = results[generateKeyFromStringUrl(tab.url)];
    if (result) {
        handleIcon(result.bias);
    }
});

async function fetchData(url: string, timeout = 8000) {
    const controller = new AbortController();
    const id = setTimeout(() => {
        setIcon("/icons/Error-icon.png");
        controller.abort();
    }, timeout);
    setIcon("/icons/loading-icon.png");

    const result = await fetch('http://52.149.65.149:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        signal: controller.signal,
        body: JSON.stringify({ url })
    });
    clearTimeout(id);
    return result;
}


function generateKeyFromStringUrl(url: string): string {
    const hostname = new URL(url).hostname;
    const domain = new URL(url).pathname;
    return `${hostname}:${domain}`;
}

function setIcon(path) {
    console.log('path');
    console.log(path);
    chrome.action.setIcon({ path });
}
async function handlePageInitialized(tabId) {
    chrome.tabs.get(tabId, async tab => {
        setIcon("/icons/scale-icon.png");
        let result = await fetchData(tab.url);
        if (!result.ok) {
            setIcon("/icons/Error-icon.png");
            return;
        }
        result = await result.json();
        console.log('parsed JSON');
        console.log(result);
        results[generateKeyFromStringUrl(tab.url)] = result;
        handleIcon((result as any).bias);
    });
}

function handleIcon(bias) {
    switch (bias) {
        case 'BALANCED':
            setIcon("/icons/Flag-green-icon.png");
            return;
        case 'MISLEADING':
        case 'TAKING_STANCE':
        case 'BIASED':
            setIcon("/icons/Flag-yellow-icon.png");
            return;
        case 'STRONGLY_BIASED':
            setIcon("/icons/Flag-red-icon.png");
            return;
    }
}
function handlePopupInitialized(pageUrl) {
    const result = results[generateKeyFromStringUrl(pageUrl)];
    if (result) {
        chrome.runtime.sendMessage({ type: 'SEND_DATA', data: results[generateKeyFromStringUrl(pageUrl)] })
    } else {
        chrome.runtime.sendMessage({ type: 'ERROR', data: "Something went wrong" });
    }
}