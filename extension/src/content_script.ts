
export type MessageType = 'PAGE_INITIALIZED' | 'POPUP_INITIALIZED' | 'SEND_DATA';

export type PageInitializedMessage = {
    type: 'PAGE_INITIALIZED'
}

export type PopupInitializedMessage = {
    type: 'POPUP_INITIALIZED',
    url: string
}

export type SendDataMessage = {
    type: 'SEND_DATA',
    data: object
}

export type Message = PageInitializedMessage | PopupInitializedMessage | SendDataMessage;

chrome.runtime.sendMessage({ type: "PAGE_INITIALIZED" });