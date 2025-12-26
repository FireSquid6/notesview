

export function encodeFilepath(fp: string): string {
  return encodeURI(fp);
}

export function decodeFilepath(fp: string): string {
  return decodeURI(fp);
}
