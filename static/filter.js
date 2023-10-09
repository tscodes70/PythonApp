$(document).ready(function () {
    // Load the dropdown with unique provinces
    $.ajax({
        url: '/get_provinces',
        type: 'GET',
        success: function (data) {
            var dropdown = $('#province-dropdown');
            dropdown.empty();
            dropdown.append($('<option></option>').attr('value', '').text('All'));
            $.each(data, function (key, entry) {
                dropdown.append($('<option></option>').attr('value', entry).text(entry));
            });
        }
    });

    // Attach an event listener to the dropdown to filter data when a province is selected
    $('#province-dropdown').on('change', function () {
        var selectedProvince = $(this).val();
        filterData(selectedProvince);
    });

    // Attach an event listener to the "Generate Word Cloud" button
    $('#generate-wordcloud').on('click', function () {
        var selectedProvince = $('#province-dropdown').val();
        generateWordCloud(selectedProvince);
    });
});

function filterData(province) {
    $.ajax({
        url: '/filter_data',
        type: 'GET',
        data: { province: province },
        success: function (data) {
            $('#filtered-data').html(data);
        }
    });
}

function generateWordCloud(province) {
    $.ajax({
        url: '/generate_wordcloud',
        type: 'GET',
        data: { province: province },
        success: function (data) {
            $('#wordcloud-container').html(data);
        }
    });
}
