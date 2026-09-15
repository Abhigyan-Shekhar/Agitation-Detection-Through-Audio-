create extension if not exists pgcrypto;

create table if not exists public.behaviour_events (
    id uuid primary key default gen_random_uuid(),
    recorded_at timestamptz not null,
    audio_timestamp text not null,
    behaviour text not null
);

create index if not exists behaviour_events_recorded_at_idx on public.behaviour_events (recorded_at desc);
alter table public.behaviour_events enable row level security;

-- No public/anon policy is created. Access is backend-only through the server-side key.
