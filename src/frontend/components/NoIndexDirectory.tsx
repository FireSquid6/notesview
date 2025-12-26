interface NoIndexDirectory {
  title: string;
}

export function NoIndexDirectory({ title }: NoIndexDirectory): JSX.Element {

  return (
    <>
      <h1>{title}</h1>
    </>
  );
}
