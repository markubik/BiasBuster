import Options from "./components/App.svelte";

type IStorage = {
    count: number;
}
chrome.storage.sync.get({ count: 0 } as IStorage, () => {
    const app = new Options({
        target: document.body,
    });
});
