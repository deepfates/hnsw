export class PriorityQueue<T> {
  private items: T[] = [];

  constructor(private compare: (a: T, b: T) => number) {}

  push(item: T) {
    let i = 0;
    while (i < this.items.length && this.compare(item, this.items[i]) > 0) {
      i++;
    }
    this.items.splice(i, 0, item);
  }

  pop(): T | undefined {
    return this.items.shift();
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }
}
