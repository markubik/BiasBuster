import type { Message } from "./content_script";

export type HateSpeechType = "NORMAL" | "OFFENSIVE" | "HATESPEECH"
export type HyperpartizanType = "NORMAL" | "HYPERPARTIZAN"
export type StanceType = "UNRELATED" | "AGREE" | "DISAGREE" | "DISCUSS"

export interface Bias {
    bias: 'UNBIASED' | 'BIASED' | 'STRONGLY_BIASED'
    predictions: {
        hatespeech: HateSpeechType,
        hyperpartizan: HyperpartizanType,
        stance: StanceType
    }
}

const results = {};

chrome.tabs.onActivated.addListener(activeInfo =>
    chrome.runtime.onMessage.addListener(
        (message: Message) => {
            if (message.type === 'PAGE_INITIALIZED') {
                handlePageInitialized(activeInfo.tabId);
            }
            else if (message.type === 'POPUP_INITIALIZED') {
                handlePopupInitialized(message.url);
            }
        }));


async function fetchData(url: string, timeout = 8000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    chrome.action.setIcon({ path: "/icons/cat-icon.png" });
    const result = await fetch('http://localhost:5000/resource', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        signal: controller.signal,
        body: JSON.stringify({ message: url })
    });
    clearTimeout(id);
    return result;
}

async function handlePageInitialized(tabId) {
    chrome.tabs.get(tabId, async tab => {
        let result = await fetchData(tab.url);
        result = await result.json();
        results[new URL(tab.url).hostname] = result;
        chrome.action.setIcon({ path: "/icons/Flag-green-icon.png" });
    });
}

function handlePopupInitialized(pageUrl) {
    const url = new URL(pageUrl);
    chrome.runtime.sendMessage({ type: 'SEND_DATA', data: results[url.hostname] })
}