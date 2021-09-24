import type { Message } from "./content_script";


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


function fetchData(url: string) {
    chrome.action.setIcon({ path: "/icons/cat-icon.png" });
    return fetch('http://localhost:5000/resource', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({ message: url })
    })
}

function handlePageInitialized(tabId) {
    chrome.tabs.get(tabId, tab =>
        fetchData(tab.url)
            .then(res => res.json())
            .then(res => {
                results[new URL(tab.url).hostname] = res;
                console.log('dostalismy dane');
                console.log(res);
                chrome.action.setIcon({ path: "/icons/Flag-green-icon.png" });
            })
    );
}

function handlePopupInitialized(pageUrl) {
    const url = new URL(pageUrl);
    console.log('pa tera');
    console.log(results[url.hostname]);
    chrome.runtime.sendMessage({ type: 'SEND_DATA', data: results[url.hostname] })
}