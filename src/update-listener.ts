import fs from "fs";

export type Subscriber = () => void | Promise<void>;

export class UpdateListener {
  private rootDirectory: string;
  private subscribers: Subscriber[] = [];
  constructor(dir: string) {
    this.rootDirectory = dir;

    fs.watch(this.rootDirectory, { recursive: true }, () => {
      this.onUpdate();
    })
  }

  private async onUpdate() {
    await Promise.all(this.subscribers.map(s => s()));
  }

  subscribe(subscriber: Subscriber): () => void {
    this.subscribers.push(subscriber);
    return () => {
      this.unsubscribe(subscriber);
    }
  }

  unsubscribe(subscriber: Subscriber) {
    this.subscribers = this.subscribers.filter(s => s !== subscriber);
  }
}
