# Incident response

Synthetic document describing the fictional Northwind Mutual runbook.

## Severity levels

Severity one covers loss of customer funds, exposure of personal data, or a complete outage of
the payments path. Severity two covers degraded service with a workaround. Severity three
covers everything else. The severity assigned at declaration may be raised later but is never
lowered retroactively, because a lowered severity rewrites the record of what was known at the
time.

## Declaration

Any engineer may declare an incident. Declaration does not require approval, and no one may be
questioned for declaring one that turns out to be minor. The failure mode this guards against
is hesitation, which costs far more than a false alarm.

## Regulator notification

A severity one incident involving personal data is reportable to the supervisory authority
within seventy-two hours of detection. The clock starts at detection, not at confirmation.
Notification includes the categories of data affected, the approximate number of records, and
the remediation timeline.

## Post-incident review

A written review is due within ten working days. It records the timeline, the contributing
factors, and the actions taken. It does not name individuals, because a review that assigns
personal blame stops producing accurate timelines within about two cycles.
