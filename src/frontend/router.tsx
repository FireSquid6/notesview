import React, { createContext, useContext, useEffect, useState } from "react";

const routerContext = createContext<{
  path: string;
  navigate: (to: string) => void;
} | null>(null);

export function Router({ children }: { children: React.ReactNode }) {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const handlePopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigate = (to: string) => {
    window.history.pushState({}, "", to);
    setPath(to);
  };

  return (
    <routerContext.Provider value={{ path, navigate }}>
      {children}
    </routerContext.Provider>
  );
}

export function Route({ path, children }: { path: string; children: React.ReactNode }) {
  const context = useContext(routerContext);
  if (!context) throw new Error("Route must be used within Router");
  
  return context.path === path ? <>{children}</> : null;
}

export function Link({ to, className, children }: { to: string; className: string; children: React.ReactNode }) {
  const context = useContext(routerContext);
  if (!context) throw new Error("Link must be used within Router");

  return (
    <a
      href={to}
      className={className}
      onClick={(e) => {
        e.preventDefault();
        context.navigate(to);
      }}
    >
      {children}
    </a>
  );
}

export function usePath(): string {
  const ctx = useContext(routerContext);

  return ctx?.path ?? "/";
}
