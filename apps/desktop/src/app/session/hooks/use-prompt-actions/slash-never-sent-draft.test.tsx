import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { setSessions } from '@/store/session'
import { dropSessionState } from '@/store/session-states'

import type { CreateBackendSessionForSend } from '../use-session-actions/create-overrides'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID, sessionInfo } from './index-test-utils'
import { SubmissionDestinationChangedError } from './submission-destination'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

let startingProfile = ''

beforeEach(() => {
  startingProfile = $activeGatewayProfile.get()
  setSessions(() => [sessionInfo()])
})

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set(startingProfile)
  dropSessionState(RUNTIME_SESSION_ID)
  setSessions(() => [])
})

async function submitFromFreshChat(
  createBackendSessionForSend: CreateBackendSessionForSend,
  draft = `/ultrathink ${'a long prompt the user spent minutes writing. '.repeat(40)}`,
  requestGateway = vi.fn(async (_method: string) => ({}) as never)
) {
  const onComposerScopeAssigned = vi.fn()

  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      activeSessionId={null}
      createBackendSessionForSend={createBackendSessionForSend}
      onReady={value => {
        handle = value
      }}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
      storedSessionId={null}
    />
  )

  let submitted: boolean | undefined

  await act(async () => {
    submitted = await handle!.submitTextRaw(draft, { onComposerScopeAssigned })
  })

  return { onComposerScopeAssigned, requestGateway, submitted }
}

it('returns the draft when the destination guard refuses the command after a fresh chat is created', async () => {
  // Creating the chat moves the ambient destination the command was captured against, so the
  // guard refuses slash.exec and its command.dispatch fallback before either leaves the renderer.
  const createBackendSessionForSend = vi.fn<CreateBackendSessionForSend>(async (_preview, overrides) => {
    $activeGatewayProfile.set(`${startingProfile || 'default'}-moved`)
    overrides?.onComposerScopeAssigned?.('stored-fresh-chat')

    return RUNTIME_SESSION_ID
  })

  const { onComposerScopeAssigned, requestGateway, submitted } = await submitFromFreshChat(createBackendSessionForSend)

  expect(submitted).toBe(false)
  expect(onComposerScopeAssigned).toHaveBeenCalledWith('stored-fresh-chat')
  // The command never reached the backend, so a restored retry cannot run it twice.
  expect(requestGateway.mock.calls.some(([method]) => method === 'slash.exec' || method === 'command.dispatch')).toBe(
    false
  )
})

it('returns the draft when the fresh chat for the command cannot be created', async () => {
  const createBackendSessionForSend = vi.fn<CreateBackendSessionForSend>(async () => null)

  const { requestGateway, submitted } = await submitFromFreshChat(createBackendSessionForSend)

  expect(submitted).toBe(false)
  expect(requestGateway.mock.calls.some(([method]) => method === 'slash.exec' || method === 'command.dispatch')).toBe(
    false
  )
})

it('returns the /btw question when the destination guard refuses prompt.btw', async () => {
  const createBackendSessionForSend = vi.fn<CreateBackendSessionForSend>(async () => {
    $activeGatewayProfile.set(`${startingProfile || 'default'}-moved`)

    return RUNTIME_SESSION_ID
  })

  const { requestGateway, submitted } = await submitFromFreshChat(
    createBackendSessionForSend,
    '/btw what did we decide about the team builder pricing?'
  )

  expect(submitted).toBe(false)
  expect(requestGateway.mock.calls.some(([method]) => method === 'prompt.btw')).toBe(false)
})

it.each([
  { case: 'slash.exec refused locally, command.dispatch timed out', exec: 'guard', dispatch: 'timeout' },
  { case: 'slash.exec timed out, command.dispatch refused locally', exec: 'timeout', dispatch: 'guard' }
])('keeps the send when only one route is known not to have run ($case)', async ({ exec, dispatch }) => {
  const failure = (kind: string) =>
    kind === 'guard' ? new SubmissionDestinationChangedError() : new Error('slash worker timed out')

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'slash.exec') {
      throw failure(exec)
    }

    if (method === 'command.dispatch') {
      throw failure(dispatch)
    }

    return {} as never
  })

  const createBackendSessionForSend = vi.fn<CreateBackendSessionForSend>(async () => RUNTIME_SESSION_ID)

  const { submitted } = await submitFromFreshChat(createBackendSessionForSend, '/ultrathink keep me', requestGateway)

  // The timed-out route may have run the command: a restored retry could run it twice.
  expect(submitted).toBe(true)
})
