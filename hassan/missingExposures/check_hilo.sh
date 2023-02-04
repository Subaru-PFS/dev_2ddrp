psql -h 133.40.152.102 -p 5433 -U pfs -d registry_gen2 -c "select distinct to_hex(pfsDesignId) from raw where (visit between 84574 and 84578
) OR (visit between 84580 and 84632) OR (visit between 84666 and 84695) OR (visit between 84720 and 84737) OR (visit between 84742 and 84753) OR (visit between 84763 and 84792) OR (visit between 84795 and 84824) OR (visit between 84827 and 84886) OR (visit between 84905 and 84934);"


